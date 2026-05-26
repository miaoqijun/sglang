"""Registry for client-defined prompt templates used by TemplateAwareChunkCache.

A prompt template is an ordered list of segments, alternating fixed text and
named variable placeholders. The TokenizerManager pre-tokenizes the fixed
segments at registration time. At request time, it expands the template by
re-tokenizing each variable segment individually and concatenating the segment
token lists, while emitting (segment_boundaries, segment_kinds) so the cache
can carve KV by template segment.

Note: concatenating per-segment tokenizations is not always byte-for-byte
identical to tokenizing the fully assembled text (BPE merges across the
boundary). This prototype accepts that drift in exchange for the
direct chunk-to-segment alignment the cache needs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SEGMENT_KIND_FIXED = "fixed"
SEGMENT_KIND_VAR = "var"


@dataclass
class TemplateSegment:
    kind: str  # SEGMENT_KIND_FIXED or SEGMENT_KIND_VAR
    text: Optional[str] = None  # set when kind == "fixed"
    var_name: Optional[str] = None  # set when kind == "var"

    def __post_init__(self):
        if self.kind == SEGMENT_KIND_FIXED:
            if self.text is None:
                raise ValueError("fixed segment must specify text")
        elif self.kind == SEGMENT_KIND_VAR:
            if not self.var_name:
                raise ValueError("var segment must specify var_name")
        else:
            raise ValueError(
                f"Unknown segment kind {self.kind!r}; expected 'fixed' or 'var'"
            )


@dataclass
class PromptTemplate:
    template_id: str
    segments: List[TemplateSegment]
    # seg_idx -> pre-tokenized tokens; populated for fixed segments at register time.
    fixed_segment_tokens: Dict[int, List[int]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PromptTemplate":
        template_id = payload.get("template_id")
        if not template_id:
            raise ValueError("template_id is required")
        raw_segments = payload.get("segments") or []
        if not raw_segments:
            raise ValueError("template must contain at least one segment")
        segments: List[TemplateSegment] = []
        for raw in raw_segments:
            segments.append(
                TemplateSegment(
                    kind=raw.get("kind", SEGMENT_KIND_FIXED),
                    text=raw.get("text"),
                    var_name=raw.get("var_name"),
                )
            )
        return cls(template_id=template_id, segments=segments)


class PromptTemplateRegistry:
    """Thread-safe registry of registered prompt templates."""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._lock = threading.Lock()

    def register(self, template: PromptTemplate, tokenizer: Any) -> PromptTemplate:
        with self._lock:
            for idx, seg in enumerate(template.segments):
                if seg.kind == SEGMENT_KIND_FIXED:
                    tokens = _encode_no_special_tokens(tokenizer, seg.text or "")
                    template.fixed_segment_tokens[idx] = tokens
            self._templates[template.template_id] = template
            return template

    def get(self, template_id: str) -> Optional[PromptTemplate]:
        with self._lock:
            return self._templates.get(template_id)

    def __contains__(self, template_id: str) -> bool:
        with self._lock:
            return template_id in self._templates


def _encode_no_special_tokens(tokenizer: Any, text: str) -> List[int]:
    """Encode without BOS/EOS or any other special tokens; concatenation friendly."""
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def expand_template(
    template: PromptTemplate,
    template_vars: Dict[str, str],
    tokenizer: Any,
) -> Tuple[List[int], List[int], List[str]]:
    """Concatenate per-segment tokens; return (input_ids, segment_boundaries, segment_kinds).

    segment_boundaries has length len(segments)+1. Segment i covers
    input_ids[boundaries[i]:boundaries[i+1]].
    """
    input_ids: List[int] = []
    boundaries: List[int] = [0]
    kinds: List[str] = []
    template_vars = template_vars or {}
    for idx, seg in enumerate(template.segments):
        if seg.kind == SEGMENT_KIND_FIXED:
            seg_tokens = template.fixed_segment_tokens.get(idx)
            if seg_tokens is None:
                # Fallback: tokenize on the fly if registration didn't precompute.
                seg_tokens = _encode_no_special_tokens(tokenizer, seg.text or "")
        else:
            if seg.var_name not in template_vars:
                raise KeyError(
                    f"template {template.template_id!r} expects var {seg.var_name!r}"
                )
            seg_tokens = _encode_no_special_tokens(
                tokenizer, str(template_vars[seg.var_name])
            )
        input_ids.extend(seg_tokens)
        boundaries.append(len(input_ids))
        kinds.append(seg.kind)
    return input_ids, boundaries, kinds
