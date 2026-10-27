import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from qwen3_asr_model.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRThinkerConfig
from qwen3_asr_model.modeling_qwen3_asr import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRAudioEncoderLayer,
    Qwen3ASRCTCBranch,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRThinkerForConditionalGeneration,
)


class AudioFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.arange(96, dtype=torch.float32).reshape(2, 8, 6)
        self.thinker = SimpleNamespace(
            audio_tower=Mock(
                side_effect=lambda features, **kwargs: SimpleNamespace(
                    last_hidden_state=features.T
                )
            )
        )

    def test_explicit_lengths_work_without_attention_mask(self):
        combined, per_audio = Qwen3ASRThinkerForConditionalGeneration.get_audio_features(
            self.thinker,
            self.features,
            audio_feature_lengths=torch.tensor([3, 5]),
            return_per_audio=True,
        )
        self.assertEqual([tuple(f.shape) for f in per_audio], [(3, 8), (5, 8)])
        torch.testing.assert_close(
            combined, torch.cat([self.features[0, :, :3].T, self.features[1, :, :5].T])
        )

    def test_attention_mask_takes_precedence_over_explicit_lengths(self):
        combined = Qwen3ASRThinkerForConditionalGeneration.get_audio_features(
            self.thinker,
            self.features,
            feature_attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]),
            audio_feature_lengths=torch.tensor([6, 6]),
        )
        torch.testing.assert_close(
            combined, torch.cat([self.features[0, :, :3].T, self.features[1, :, :5].T])
        )

    def test_missing_length_metadata_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "feature_attention_mask.*audio_feature_lengths"):
            Qwen3ASRThinkerForConditionalGeneration.get_audio_features(
                self.thinker, self.features
            )


class CTCLossTest(unittest.TestCase):
    def compute_loss(self, logits, lengths, labels, label_lengths):
        thinker = SimpleNamespace(
            ctc_branch=SimpleNamespace(blank_id=0),
            _ctc_branch_packed=lambda features: (logits, torch.tensor(lengths)),
        )
        return Qwen3ASRThinkerForConditionalGeneration._ctc_loss_from_feats(
            thinker,
            [torch.zeros(length, 2) for length in lengths],
            torch.tensor(labels),
            torch.tensor(label_lengths),
        )

    def test_target_longer_than_audio_has_zero_loss_and_gradient(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        loss = self.compute_loss(logits, [2], [[1, 2, 3]], [3])
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_repeated_labels_without_enough_frames_have_zero_loss(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        loss = self.compute_loss(logits, [2], [[1, 1]], [2])
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_invalid_sample_does_not_learn_from_padding_in_mixed_batch(self):
        logits = torch.zeros(6, 4, requires_grad=True)
        loss = self.compute_loss(logits, [2, 4], [[1, 2, 3], [1, 2, 3]], [3, 3])
        valid_loss = F.ctc_loss(
            logits[2:].log_softmax(-1).unsqueeze(1),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([4]),
            torch.tensor([3]),
            blank=0,
            zero_infinity=True,
        )
        torch.testing.assert_close(loss, valid_loss / 2)
        loss.backward()
        torch.testing.assert_close(logits.grad[:2], torch.zeros_like(logits[:2]))
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad[2:].abs().sum().item(), 0)

    def test_valid_variable_length_batch_matches_individual_losses(self):
        logits = torch.linspace(-1, 1, 32).reshape(8, 4).requires_grad_()
        loss = self.compute_loss(logits, [3, 5], [[1, 2], [2, 3]], [2, 2])
        individual_losses = []
        for frames, target in ((logits[:3], [1, 2]), (logits[3:], [2, 3])):
            individual_losses.append(
                F.ctc_loss(
                    frames.log_softmax(-1).unsqueeze(1),
                    torch.tensor([target]),
                    torch.tensor([len(frames)]),
                    torch.tensor([len(target)]),
                    blank=0,
                    zero_infinity=True,
                )
            )
        torch.testing.assert_close(loss, torch.stack(individual_losses).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())


class GenerationTest(unittest.TestCase):
    def test_generation_and_model_arguments_reach_thinker(self):
        delegated = Mock(return_value=object())
        model = SimpleNamespace(thinker=SimpleNamespace(generate=delegated))
        input_ids = torch.tensor([[1]])
        extra = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "input_features": torch.zeros(1, 8, 100),
            "feature_attention_mask": torch.ones(1, 100, dtype=torch.long),
            "audio_feature_lengths": torch.tensor([100]),
        }
        result = Qwen3ASRForConditionalGeneration.generate(
            model,
            input_ids=input_ids,
            max_new_tokens=32,
            repetition_penalty=1.2,
            num_beams=2,
            eos_token_id=[2],
            **extra,
        )
        self.assertIs(result, delegated.return_value)
        forwarded = delegated.call_args.kwargs
        self.assertIs(forwarded["input_ids"], input_ids)
        for key, value in extra.items():
            self.assertIs(forwarded[key], value)
        self.assertEqual(forwarded["max_new_tokens"], 32)
        self.assertEqual(forwarded["repetition_penalty"], 1.2)
        self.assertEqual(forwarded["num_beams"], 2)
        self.assertEqual(forwarded["eos_token_id"], [2])
        self.assertTrue(forwarded["return_dict_in_generate"])

    def test_explicit_tensor_output_is_respected(self):
        sequences = torch.tensor([[1, 2]])
        delegated = Mock(return_value=sequences)
        model = SimpleNamespace(thinker=SimpleNamespace(generate=delegated))
        result = Qwen3ASRForConditionalGeneration.generate(
            model, input_ids=torch.tensor([[1]]), return_dict_in_generate=False
        )
        self.assertIs(result, sequences)
        self.assertFalse(delegated.call_args.kwargs["return_dict_in_generate"])


class CTCDropoutTest(unittest.TestCase):
    @staticmethod
    def make_config(dropout, backend):
        config = Qwen3ASRThinkerConfig(
            audio_config={"d_model": 8, "output_dim": 8, "encoder_attention_heads": 2,
                          "encoder_ffn_dim": 16},
            ctc_num_layers=1, ctc_num_heads=2, ctc_ffn_dim=16, ctc_vocab_size=10,
            ctc_dropout=dropout,
        )
        config.audio_config._attn_implementation = backend
        return config

    @torch.no_grad()
    def test_ctc_dropout_applies_in_training_and_is_disabled_in_eval(self):
        features = torch.randn(12, 8)
        lengths = torch.tensor([0, 12], dtype=torch.int32)
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                branch = Qwen3ASRCTCBranch(self.make_config(0.8, backend))
                mask = branch.build_attention_mask(features, lengths)
                branch.train()
                first = branch(features, lengths, attention_mask=mask)
                second = branch(features, lengths, attention_mask=mask)
                self.assertFalse(torch.equal(first, second))
                branch.eval()
                torch.testing.assert_close(
                    branch(features, lengths, attention_mask=mask),
                    branch(features, lengths, attention_mask=mask), rtol=0, atol=0,
                )

    @torch.no_grad()
    def test_pretrained_audio_layer_keeps_original_no_dropout_behavior(self):
        config = self.make_config(0.8, "eager").audio_config
        config.dropout = config.attention_dropout = config.activation_dropout = 0.8
        layer = Qwen3ASRAudioEncoderLayer(config).train()
        features = torch.randn(12, 8)
        lengths = torch.tensor([0, 12], dtype=torch.int32)
        torch.testing.assert_close(layer(features, lengths)[0], layer(features, lengths)[0], rtol=0, atol=0)


class AudioAttentionTest(unittest.TestCase):
    @staticmethod
    def make_encoder(backend):
        config = Qwen3ASRAudioEncoderConfig(
            num_mel_bins=8,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            d_model=8,
            output_dim=8,
            downsample_hidden_size=2,
            n_window=50,
            n_window_infer=100,
            max_source_positions=64,
        )
        config._attn_implementation = backend
        encoder = Qwen3ASRAudioEncoder(config).eval()
        # Avoid near-zero initialization hiding attention leaking between windows.
        for module in encoder.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.normal_(module.weight, std=0.2)
        return encoder

    @torch.no_grad()
    def test_attention_windows_match_independent_encoding(self):
        torch.manual_seed(0)
        features = torch.randn(8, 200)
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                encoder = self.make_encoder(backend)
                combined = encoder(features, feature_lens=torch.tensor([200])).last_hidden_state
                independent = torch.cat(
                    [
                        encoder(window, feature_lens=torch.tensor([100])).last_hidden_state
                        for window in features.split(100, dim=-1)
                    ]
                )
                torch.testing.assert_close(combined, independent, atol=1e-7, rtol=1e-5)

    @torch.no_grad()
    def test_eager_and_sdpa_agree(self):
        eager = self.make_encoder("eager")
        sdpa = self.make_encoder("sdpa")
        sdpa.load_state_dict(eager.state_dict())
        features = torch.randn(8, 200)
        eager_output = eager(features, feature_lens=torch.tensor([200])).last_hidden_state
        sdpa_output = sdpa(features, feature_lens=torch.tensor([200])).last_hidden_state
        torch.testing.assert_close(eager_output, sdpa_output, atol=1e-7, rtol=1e-5)

    @torch.no_grad()
    def test_flash_attention_keeps_varlen_path_without_dense_mask(self):
        encoder = self.make_encoder("eager")
        encoder.config._attn_implementation = "flash_attention_2"
        # Exercise mask routing without requiring a GPU or the FlashAttention extension.
        with patch.object(
            encoder.layers[0], "forward", side_effect=lambda hidden, *args, **kwargs: (hidden,)
        ) as layer:
            encoder(torch.randn(8, 200), feature_lens=torch.tensor([200]))
        self.assertIsNone(layer.call_args.kwargs["attention_mask"])


if __name__ == "__main__":
    unittest.main()
