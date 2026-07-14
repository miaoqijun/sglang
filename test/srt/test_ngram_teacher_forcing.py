from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.eagle_utils import apply_target_predict_override
from sglang.srt.speculative.ngram_worker import (
    NGRAM_TEACHER_FORCING_TOKEN_IDS,
    build_ngram_teacher_forcing_targets,
)


def _req(origin_len, tokens=None):
    custom_params = (
        {NGRAM_TEACHER_FORCING_TOKEN_IDS: tokens} if tokens is not None else None
    )
    return SimpleNamespace(
        origin_input_ids=list(range(origin_len)),
        sampling_params=SimpleNamespace(custom_params=custom_params),
    )


def test_build_teacher_forcing_targets_uses_absolute_tree_positions():
    reqs = [_req(3, [10, 11, 12]), _req(2)]
    positions = torch.tensor([[3, 4, 4, 5, 6], [2, 3, 4, 5, 6]])

    targets = build_ngram_teacher_forcing_targets(reqs, positions, vocab_size=32)

    assert targets.tolist() == [
        [10, 11, 11, 12, -1],
        [-1, -1, -1, -1, -1],
    ]


def test_build_teacher_forcing_targets_shifts_verify_logits_to_next_token():
    reqs = [_req(3, [10, 11, 12])]
    positions = torch.tensor([[3, 4, 4, 5, 6]])

    targets = build_ngram_teacher_forcing_targets(
        reqs, positions, vocab_size=32, position_shift=1
    )

    assert targets.tolist() == [[11, 12, 12, -1, -1]]


def test_build_teacher_forcing_targets_accepts_overlap_output_offsets():
    reqs = [_req(20, [10, 11, 12, 13])]
    positions = torch.tensor([[20, 21, 21, 22]])

    targets = build_ngram_teacher_forcing_targets(
        reqs, positions, vocab_size=32, output_offsets=[1]
    )

    assert targets.tolist() == [[11, 12, 12, 13]]


def test_build_teacher_forcing_targets_validates_tokens():
    with pytest.raises(ValueError, match="invalid token id"):
        build_ngram_teacher_forcing_targets(
            [_req(1, [7])], torch.tensor([1]), vocab_size=7
        )

    with pytest.raises(ValueError, match="integer token ids"):
        build_ngram_teacher_forcing_targets(
            [_req(1, [1.5])], torch.tensor([1]), vocab_size=7
        )


def test_target_predict_override_preserves_unforced_positions():
    predicted = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    override = torch.tensor([[9, -1, 8], [-1, -1, 7]], dtype=torch.int32)

    result = apply_target_predict_override(predicted, override)

    assert result.tolist() == [[9, 2, 8], [4, 5, 7]]


def test_target_predict_override_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        apply_target_predict_override(torch.zeros(2, 3), torch.zeros(3, 2))
