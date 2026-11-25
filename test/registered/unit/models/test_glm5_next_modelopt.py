import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration, Glm5NextMLP
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NVIDIA_EXCLUDE_MODULES = [
    "model.language_model.embed_tokens",
    "model.language_model.layers.11.self_attn*",
    "model.language_model.layers.11.mlp.gate",
    "model.language_model.layers.11.mlp.shared_experts*",
    "model.visual*",
]

RADIXARK_EXCLUDE_MODULES = [
    "model.language_model.embed_tokens",
    "model.language_model.layers.11.self_attn*",
    "model.language_model.layers.11.mlp.gate",
    "model.visual*",
    "model.embed_tokens",
    "model.layers.11.self_attn*",
    "model.layers.11.mlp.gate",
    "model.language_model.layers.45*",
    "model.layers.45*",
    "visual*",
]


class TestGlm5NextModelOpt(CustomTestCase):
    def _config(self, exclude_modules):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=exclude_modules,
        )
        config.apply_weight_name_mapper(
            Glm5NextForConditionalGeneration.hf_to_sglang_mapper
        )
        return config

    def _hf_config(self):
        return SimpleNamespace(
            text_config=SimpleNamespace(
                first_k_dense_replace=3,
                num_hidden_layers=45,
                n_shared_experts=1,
            )
        )

    def test_checkpoint_exclusions_match_sglang_module_names(self):
        checkpoints = {
            "nvidia/GLM-5.3-Flash-NVFP4": NVIDIA_EXCLUDE_MODULES,
            "RadixArk/GLM-5.3-Flash-NVFP4": RADIXARK_EXCLUDE_MODULES,
        }

        for checkpoint, exclude_modules in checkpoints.items():
            with self.subTest(checkpoint=checkpoint):
                config = self._config(exclude_modules)
                self.assertTrue(config.is_layer_excluded("model.embed_tokens"))
                self.assertTrue(
                    config.is_layer_excluded("model.layers.11.self_attn.kv_b_proj")
                )
                self.assertTrue(config.is_layer_excluded("model.layers.11.mlp.gate"))
                self.assertTrue(
                    config.is_layer_excluded("visual.blocks.0.attn.qkv_proj")
                )

    def test_nvidia_mixed_precision_shared_experts_disable_fusion(self):
        config = self._config(NVIDIA_EXCLUDE_MODULES)

        reason = Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
            self._hf_config(), config
        )

        self.assertIn("shared experts unquantized", reason)

    def test_radixark_uniform_fp4_shared_experts_keep_fusion(self):
        config = self._config(RADIXARK_EXCLUDE_MODULES)
        a2a_backend = SimpleNamespace(is_deepep=lambda: False)

        with (
            patch.object(glm5_next, "_is_cuda", True),
            patch.object(glm5_next, "_device_sm", 100),
            patch.object(glm5_next, "get_moe_a2a_backend", return_value=a2a_backend),
            get_parallel().override(moe_ep_size=1),
        ):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    self._hf_config(), config
                )
            )

        self.assertIsNone(reason)

    def test_nextn_mapper_routes_checkpoint_names_to_runtime_modules(self):
        """Draft quant lookups miss unless checkpoint names reach NextN module paths."""
        mapper = Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(
            self._hf_config()
        )
        cases = {
            "model.language_model.layers.45.mlp.experts.7.up_proj": "model.decoder.mlp.experts.7.up_proj",
            "model.language_model.layers.45.eh_proj.weight": "model.eh_proj.weight",
            "model.layers.45.eh_proj.weight": "model.eh_proj.weight",
            "model.language_model.layers.45.enorm.weight": "model.enorm.weight",
            "model.language_model.layers.44.mlp.experts.3.up_proj": "model.layers.44.mlp.experts.3.up_proj",
            "model.visual.blocks.0.attn.proj": "visual.blocks.0.attn.proj",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(mapper._map_name(source), expected)


class TestGlm5NextNextNScaleLoading(CustomTestCase):
    def _load(self, weights, params):
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_nextn_predict_layers=1, num_hidden_layers=45, n_routed_experts=2
            ),
            num_fused_shared_experts=0,
            fuse_qkv_a_proj=False,
            quant_config=None,
            named_parameters=lambda: params.items(),
            model=SimpleNamespace(decoder=SimpleNamespace(self_attn=None)),
        )
        weights = [(f"model.language_model.layers.45.{k}", v) for k, v in weights]
        Glm5NextForConditionalGenerationNextN.load_weights(model, weights)

    def _param(self):
        param = torch.nn.Parameter(torch.full((1, 2), -1.0), requires_grad=False)
        param.weight_loader = Mock(
            side_effect=lambda p, w, *args, **kwargs: p.data.copy_(w)
        )
        return param

    def test_existing_scale_parameter_takes_priority(self):
        for source, target in (
            ("self_attn.q_b_proj", "self_attn.q_b_proj."),
            ("mlp.experts.0.gate_proj", "mlp.experts.w13_"),
        ):
            with self.subTest(source=source):
                name = f"model.decoder.{target}weight_scale"
                param, unused = self._param(), self._param()
                scale = torch.tensor([[0.25, 0.5]])
                self._load(
                    [(f"{source}.weight_scale", scale)],
                    {name: param, f"{name}_inv": unused},
                )
                torch.testing.assert_close(param, scale)
                unused.weight_loader.assert_not_called()


class TestGlm5NextNextNModelOptFp4Resolution(CustomTestCase):
    def _make_config(self, exclude_modules):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=exclude_modules,
        )
        config.apply_weight_name_mapper(
            Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(
                SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=45))
            )
        )
        return config

    def test_preserves_modelopt_fp4_when_checkpoint_quantizes_nextn_experts(self):
        from sglang.srt.models import glm5_next_nextn

        config = self._make_config(
            ["*.self_attn.*", "*.mlp.shared_experts.*", "*.mlp.gate"]
        )
        model = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )
        model._root_quantization_config = {"ignore": ["*.self_attn.*"]}
        text_cfg = SimpleNamespace(num_hidden_layers=45)
        fake_keys = {
            "model.language_model.layers.45.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale",
            "model.language_model.layers.45.self_attn.q_b_proj.weight",
            "model.language_model.layers.45.self_attn.o_proj.weight",
            "model.language_model.layers.45.mlp.shared_experts.down_proj.weight",
        }

        with patch.object(
            glm5_next_nextn, "_load_checkpoint_nextn_keys", return_value=fake_keys
        ):
            resolved = model._resolve_nextn_quant_config(text_cfg, config)
            preserve = model._should_preserve_modelopt_fp4_nextn(text_cfg, resolved)
        self.assertIs(resolved, config)
        self.assertTrue(preserve)

    def test_drops_modelopt_fp4_when_nextn_excluded_or_unquantized(self):
        from sglang.srt.models import glm5_next_nextn

        config = self._make_config(RADIXARK_EXCLUDE_MODULES)
        model = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )
        model._root_quantization_config = {"ignore": RADIXARK_EXCLUDE_MODULES}
        text_cfg = SimpleNamespace(num_hidden_layers=45)
        fake_keys = {
            "model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale",
        }

        with patch.object(
            glm5_next_nextn, "_load_checkpoint_nextn_keys", return_value=fake_keys
        ):
            resolved = model._resolve_nextn_quant_config(text_cfg, config)
            preserve = model._should_preserve_modelopt_fp4_nextn(text_cfg, resolved)
        self.assertFalse(preserve)

    def test_drops_modelopt_fp4_when_unquantized_submodule_is_not_excluded(self):
        from sglang.srt.models import glm5_next_nextn

        # self_attn is NOT excluded, yet checkpoint has BF16 q_b_proj without weight_scale.
        config = self._make_config(["*.mlp.shared_experts.*", "*.mlp.gate"])
        model = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )
        model._root_quantization_config = {"ignore": []}
        text_cfg = SimpleNamespace(num_hidden_layers=45)
        fake_keys = {
            "model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale",
            "model.language_model.layers.45.self_attn.q_b_proj.weight",
        }

        with (
            patch.object(
                glm5_next_nextn, "_load_checkpoint_nextn_keys", return_value=fake_keys
            ),
            self.assertLogs(glm5_next_nextn.logger, level="WARNING") as logs,
        ):
            preserve = model._should_preserve_modelopt_fp4_nextn(text_cfg, config)
        self.assertFalse(preserve)
        self.assertTrue(any("self_attn.q_b_proj" in msg for msg in logs.output))

    def test_load_checkpoint_nextn_keys_supports_single_file_safetensors(self):
        from sglang.srt.models import glm5_next_nextn

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(
                {
                    "model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale": torch.zeros(
                        1, dtype=torch.float32
                    ),
                    "model.language_model.layers.44.mlp.experts.0.gate_proj.weight_scale": torch.zeros(
                        1, dtype=torch.float32
                    ),
                },
                os.path.join(tmpdir, "model.safetensors"),
            )
            with (
                patch.object(
                    glm5_next_nextn,
                    "get_spec",
                    return_value=SimpleNamespace(speculative_draft_model_path=None),
                ),
                patch.object(
                    glm5_next_nextn,
                    "get_model",
                    return_value=SimpleNamespace(model_path=tmpdir),
                ),
            ):
                keys = glm5_next_nextn._load_checkpoint_nextn_keys(45)
        self.assertEqual(
            keys,
            {"model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale"},
        )

    def test_load_checkpoint_nextn_keys_logs_warning_on_corrupt_index(self):
        from sglang.srt.models import glm5_next_nextn

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "model.safetensors.index.json"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("{invalid json")
            with (
                patch.object(
                    glm5_next_nextn,
                    "get_spec",
                    return_value=SimpleNamespace(speculative_draft_model_path=None),
                ),
                patch.object(
                    glm5_next_nextn,
                    "get_model",
                    return_value=SimpleNamespace(model_path=tmpdir),
                ),
                self.assertLogs(glm5_next_nextn.logger, level="WARNING") as logs,
            ):
                keys = glm5_next_nextn._load_checkpoint_nextn_keys(45)
        self.assertIsNone(keys)
        self.assertTrue(any("model.safetensors.index.json" in m for m in logs.output))

    def test_indexer_kpool_forward_k_only_calls_skip_logits_without_act_quant(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
        from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

        idx = IndexerKPool.__new__(IndexerKPool)
        called = {}

        def fake_skip_logits(
            x, positions, forward_batch, layer_id, metadata, return_indices=True
        ):
            called.update(
                x=x,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
                return_indices=return_indices,
            )

        idx._forward_cuda_skip_logits = fake_skip_logits
        fake_meta = object()
        fake_backend = SimpleNamespace(get_indexer_metadata=lambda lid, fb: fake_meta)
        with patch.object(
            dsa_indexer_kpool, "get_attn_backend", return_value=fake_backend
        ):
            idx.forward_k_only(
                x="x", positions="pos", forward_batch="fb", layer_id=45
            )
        self.assertEqual(called["layer_id"], 45)
        self.assertIs(called["metadata"], fake_meta)
        self.assertFalse(called["return_indices"])


class TestGlm5NextPrefillAutotune(CustomTestCase):
    def _make_fp4_linear(
        self,
        in_features: int,
        out_features: int,
        *,
        interleaved_empty: bool = False,
        accepts_prequant: bool = False,
    ):
        linear = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(linear)
        linear.input_size_per_partition = in_features
        linear.output_size_per_partition = out_features
        linear.params_dtype = torch.bfloat16
        qm = ModelOptFp4LinearMethod.__new__(ModelOptFp4LinearMethod)
        qm.quant_mode = "w4a4"
        qm.apply = Mock(
            side_effect=lambda layer, x: torch.zeros(
                (x[0].shape[0] if isinstance(x, tuple) else x.shape[0], out_features),
                dtype=torch.bfloat16,
            )
        )
        linear.quant_method = qm
        linear._interleave_for_swiglu_fusion = interleaved_empty
        linear._accepts_prequantized_fp4 = accepts_prequant
        if interleaved_empty:
            # Mirrors ModelOptFp4LinearMethod.process_weights_after_loading,
            # which empties layer.weight to a 1D tensor after interleaving.
            linear.weight = torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8), requires_grad=False
            )
            linear.weight_swiglu_interleaved = torch.zeros(
                (out_features, in_features // 2), dtype=torch.uint8
            )
            linear.weight_scale_swiglu_interleaved = torch.zeros(
                (out_features, in_features // 16), dtype=torch.uint8
            )
            linear.alpha = torch.ones(1, dtype=torch.float32)
            linear.input_scale_inv = torch.ones(1, dtype=torch.float32)
        else:
            linear.weight = torch.nn.Parameter(
                torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
                requires_grad=False,
            )
            linear.input_scale_inv = torch.ones(1, dtype=torch.float32)
        return linear

    def test_autotune_prefill_kernels_skips_interleaved_swiglu_and_tunes_fused_mlp(
        self,
    ):
        from sglang.srt.layers.quantization import fp4_utils

        mlp = Glm5NextMLP.__new__(Glm5NextMLP)
        torch.nn.Module.__init__(mlp)
        mlp._enable_nvfp4_gemm_swiglu_fusion = True
        mlp.swiglu_limit = 7.0
        mlp.gate_up_proj = self._make_fp4_linear(256, 512, interleaved_empty=True)
        mlp.down_proj = self._make_fp4_linear(256, 256, accepts_prequant=True)

        regular_proj = self._make_fp4_linear(128, 256)
        duplicate_proj = self._make_fp4_linear(128, 256)

        container = torch.nn.Module()
        container.dense_mlp = mlp
        container.proj_a = regular_proj
        container.proj_b = duplicate_proj

        model = Glm5NextForConditionalGeneration.__new__(
            Glm5NextForConditionalGeneration
        )
        torch.nn.Module.__init__(model)
        model.model = container

        fake_quant = Mock(
            return_value=(
                torch.zeros((32, 128), dtype=torch.uint8),
                torch.zeros((128, 16), dtype=torch.uint8),
            )
        )
        fake_fused = Mock(
            return_value=(
                torch.zeros((32, 128), dtype=torch.uint8),
                torch.zeros((128, 16), dtype=torch.uint8),
            )
        )
        fake_swiglu_mod = SimpleNamespace(nvfp4_gemm_swiglu_nvfp4_quant=fake_fused)

        with (
            patch.object(fp4_utils, "fp4_quantize", fake_quant),
            patch.dict(
                sys.modules,
                {
                    "sglang.kernels.ops.gemm.nvfp4_gemm_swiglu_nvfp4_quant": fake_swiglu_mod
                },
            ),
        ):
            tuned = model.autotune_prefill_kernels(32, dtype=torch.bfloat16)

        # 1 unique fused MLP + 1 unique regular NVFP4 linear shape.
        self.assertEqual(tuned, 2)
        # gate_up_proj has 1D empty weight and must NEVER be passed to quant_method.apply.
        mlp.gate_up_proj.quant_method.apply.assert_not_called()
        fake_quant.assert_called_once()
        fake_fused.assert_called_once()
        self.assertEqual(fake_fused.call_args.kwargs["swiglu_limit"], 7.0)
        # down_proj is tuned with the pre-quantized (out_fp4, out_scale) tuple.
        mlp.down_proj.quant_method.apply.assert_called_once()
        down_arg = mlp.down_proj.quant_method.apply.call_args.args[1]
        self.assertIsInstance(down_arg, tuple)
        # Regular linear tuned once; duplicate shape deduplicated.
        regular_proj.quant_method.apply.assert_called_once()
        duplicate_proj.quant_method.apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
