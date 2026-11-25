# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import logging
import os

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import get_model, get_spec

logger = logging.getLogger(__name__)


def _checkpoint_has_nvfp4_nextn_experts(layer_id: int) -> bool:
    """Return True when the draft checkpoint's index declares NVFP4 expert scales
    for the NextN layer (e.g. ``layers.45.mlp.experts.*.weight_scale``)."""
    model_path = get_spec().speculative_draft_model_path or get_model().model_path
    if not model_path or not os.path.isdir(model_path):
        return False
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
    except Exception:
        return False
    needle = f"layers.{layer_id}.mlp.experts."
    return any(
        needle in k and k.endswith(".weight_scale") for k in weight_map
    )


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    def prepare_cp_inputs(self, forward_batch, **kwargs):
        # The worker has rotated the target's multimodal embeddings per request.
        # Fill each appended token while indices still address full sequences,
        # then let the runner shard embeddings and target states together.
        input_embeds = self.model.embed_input_ids(
            forward_batch.input_ids, forward_batch
        )
        return input_embeds, forward_batch.positions, {}

    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        n = text_config.num_hidden_layers
        # lookups arrive as checkpoint and normalized names, so every rule has both forms
        draft_rules: dict[str, str] = {}
        for ckpt_prefix in (f"model.language_model.layers.{n}", f"model.layers.{n}"):
            # eh_proj/enorm/hnorm sit beside the block under `model`, not in `decoder`
            draft_rules[f"{ckpt_prefix}.eh_proj"] = "model.eh_proj"
            draft_rules[f"{ckpt_prefix}.enorm"] = "model.enorm"
            draft_rules[f"{ckpt_prefix}.hnorm"] = "model.hnorm"
            draft_rules[ckpt_prefix] = "model.decoder"
        # target rules still normalize the non-draft layers and vision tower in `exclude`
        return Glm5NextForConditionalGeneration.hf_to_sglang_mapper | WeightsMapper(
            orig_to_new_substr=draft_rules,
        )

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Mixed checkpoints list the BF16 NextN block in ``quantization_config.ignore``;
        inheriting global FP8 quantization would corrupt its QKV weights.
        Conversely, NVFP4 checkpoints that quantize NextN routed experts should
        preserve ``modelopt_fp4`` so layer ``num_hidden_layers`` loads and runs
        native NVFP4 MoE instead of falling back to BF16."""
        raw_quant_config = (
            getattr(self, "_root_quantization_config", None)
            or getattr(config, "quantization_config", None)
            or {}
        )
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        n = config.num_hidden_layers
        for nextn_layer_pattern in (
            f"model.layers.{n}.*",
            f"model.language_model.layers.{n}.*",
        ):
            if nextn_layer_pattern in ignored:
                logger.warning(
                    "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                    "using BF16 draft modules",
                    nextn_layer_pattern,
                )
                return None

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            is_excluded = hasattr(quant_config, "is_layer_excluded") and (
                quant_config.is_layer_excluded("model.decoder.mlp.experts")
                or quant_config.is_layer_excluded(f"model.layers.{n}.mlp.experts")
                or quant_config.is_layer_excluded(
                    f"model.language_model.layers.{n}.mlp.experts"
                )
            )
            if not is_excluded and _checkpoint_has_nvfp4_nextn_experts(n):
                self._preserve_modelopt_fp4_nextn = True
                logger.info(
                    "GLM5 NextN layer %d experts are NVFP4-quantized in checkpoint; "
                    "preserving modelopt_fp4 for draft MoE.",
                    n,
                )
                return quant_config

        return super()._resolve_nextn_quant_config(config, quant_config)

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        self._root_quantization_config = getattr(config, "quantization_config", None)
        self._preserve_modelopt_fp4_nextn = False
        Glm5NextForConditionalGeneration.register_modelslim_projection_aliases(
            quant_config
        )
        super().__init__(
            getattr(config, "text_config", config),
            quant_config=quant_config,
            prefix=prefix,
        )
        self.model.decoder.self_attn.dsa_decode_absorb_under_indexer = True

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        nextn_weights = (
            (name, weight)
            for name, weight in weights
            if name.startswith(layer_prefixes)
        )
        return Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights, is_nextn=True
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
