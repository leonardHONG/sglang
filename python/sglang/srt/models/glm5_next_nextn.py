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


def _resolve_local_checkpoint_dir(model_path: str) -> str | None:
    if not model_path:
        return None
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            model_path,
            local_files_only=True,
            allow_patterns=["model.safetensors.index.json", "model.safetensors"],
        )
        if local_dir and os.path.isdir(local_dir):
            return local_dir
    except Exception as exc:
        logger.debug(
            "Unable to resolve local HuggingFace cache for %r: %s",
            model_path,
            exc,
        )
    return None


def _load_checkpoint_nextn_keys(layer_id: int) -> set[str] | None:
    """Return all weight tensor names belonging to NextN ``layer_id`` from the
    checkpoint index (sharded) or safetensors header (single-file), or None if
    unavailable."""
    model_path = get_spec().speculative_draft_model_path or get_model().model_path
    local_dir = _resolve_local_checkpoint_dir(model_path)
    if local_dir is None:
        return None

    layer_needle = f"layers.{layer_id}."
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            return {k for k in weight_map if layer_needle in k}
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to read NextN checkpoint index %s; falling back to BF16 NextN: %s",
                index_path,
                exc,
            )
            return None

    single_path = os.path.join(local_dir, "model.safetensors")
    if os.path.isfile(single_path):
        try:
            from safetensors import safe_open

            with safe_open(single_path, framework="pt", device="cpu") as f:
                return {k for k in f.keys() if layer_needle in k}
        except Exception as exc:
            logger.warning(
                "Failed to read single-file NextN checkpoint header %s; "
                "falling back to BF16 NextN: %s",
                single_path,
                exc,
            )
            return None

    return None


def _nextn_submodule_is_excluded(quant_config, layer_id: int, submodule: str) -> bool:
    if not hasattr(quant_config, "is_layer_excluded"):
        return False
    return any(
        quant_config.is_layer_excluded(candidate)
        for candidate in (
            f"model.decoder.{submodule}",
            f"model.layers.{layer_id}.{submodule}",
            f"model.language_model.layers.{layer_id}.{submodule}",
        )
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

        return super()._resolve_nextn_quant_config(config, quant_config)

    def _should_preserve_modelopt_fp4_nextn(self, config, nextn_quant_config) -> bool:
        if (
            nextn_quant_config is None
            or nextn_quant_config.get_name() != "modelopt_fp4"
        ):
            return False
        n = config.num_hidden_layers
        if _nextn_submodule_is_excluded(nextn_quant_config, n, "mlp.experts"):
            return False
        nextn_keys = _load_checkpoint_nextn_keys(n)
        if not nextn_keys:
            return False
        expert_needle = f"layers.{n}.mlp.experts."
        has_nvfp4_experts = any(
            expert_needle in k and k.endswith(".weight_scale") for k in nextn_keys
        )
        if not has_nvfp4_experts:
            return False
        # Verify non-expert NextN linear sub-trees that lack NVFP4 weight_scale
        # in the checkpoint are actually excluded by quant_config; otherwise
        # preserving modelopt_fp4 would instantiate FP4 weights for BF16 tensors.
        for submodule, probe_weights in (
            ("self_attn.q_b_proj", (f"layers.{n}.self_attn.q_b_proj",)),
            ("self_attn.o_proj", (f"layers.{n}.self_attn.o_proj",)),
            (
                "mlp.shared_experts.down_proj",
                (f"layers.{n}.mlp.shared_experts.down_proj",),
            ),
        ):
            has_weight = any(
                any(p in k for p in probe_weights) and k.endswith(".weight")
                for k in nextn_keys
            )
            has_scale = any(
                any(p in k for p in probe_weights) and k.endswith(".weight_scale")
                for k in nextn_keys
            )
            if (
                has_weight
                and not has_scale
                and not _nextn_submodule_is_excluded(nextn_quant_config, n, submodule)
            ):
                logger.warning(
                    "GLM5 NextN layer %d has unquantized %s in checkpoint without "
                    "matching quant_config exclusion; falling back to BF16 NextN.",
                    n,
                    submodule,
                )
                return False
        logger.info(
            "GLM5 NextN layer %d experts are NVFP4-quantized in checkpoint; "
            "preserving modelopt_fp4 for draft MoE.",
            n,
        )
        return True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        self._root_quantization_config = getattr(config, "quantization_config", None)
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
