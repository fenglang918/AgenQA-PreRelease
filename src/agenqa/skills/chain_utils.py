"""Chain utilities for K/Q/A growth data.

Provides program-friendly helpers to:
- collect step-wise nodes per paper (linked-list style),
- compose Known_i from K0 + accumulated A_{j} (j < i),
- verify whether a materialized Known_i includes all prior answers,
- extract head–tail views.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import json
import logging
import re

from infra.data.io import read_jsonl


logger = logging.getLogger(__name__)


@dataclass
class KQANode:
    paper_id: str
    step: int
    known: str
    question: str
    answer: str
    subject: Optional[str] = None
    chain: Optional[str] = None
    prev_step: Optional[int] = None
    known_derivation: Optional[Dict[str, Any]] = None
    components: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    service_id: Optional[str] = None
    timestamp: Optional[str] = None


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _as_node(obj: Mapping[str, Any]) -> Optional[KQANode]:
    pid = obj.get("paper_id")
    step = _to_int(obj.get("step"))
    known = obj.get("known") or obj.get("Known")
    # Strict: expect 'question'
    question = obj.get("question")
    answer = obj.get("answer") or obj.get("Answer")
    if not (isinstance(pid, str) and isinstance(step, int) and isinstance(known, str) and isinstance(question, str) and isinstance(answer, str)):
        return None
    return KQANode(
        paper_id=pid,
        step=step,
        known=known,
        question=question,
        answer=answer,
        subject=(obj.get("subject") if isinstance(obj.get("subject"), str) else None),
        chain=obj.get("chain"),
        prev_step=_to_int(obj.get("prev_step")),
        known_derivation=(obj.get("known_derivation") if isinstance(obj.get("known_derivation"), dict) else None),
        components=(obj.get("components") if isinstance(obj.get("components"), dict) else None),
        model=(obj.get("model") if isinstance(obj.get("model"), str) else None),
        service_id=(obj.get("service_id") if isinstance(obj.get("service_id"), str) else None),
        timestamp=(obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None),
    )


def read_nodes(path: Path) -> List[KQANode]:
    """Read nodes from JSONL or consecutive JSON objects."""
    # try stream-concatenated JSON first
    text: str
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        # fallback JSONL
        return [n for n in (_as_node(obj) for obj in read_jsonl(path, schema=None)) if n]

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    nodes: List[KQANode] = []
    yielded = 0
    while True:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                yielded += 1
                node = _as_node(obj)
                if node:
                    nodes.append(node)
            idx = end
        except json.JSONDecodeError:
            yielded = 0
            break

    if yielded == 0:
        return [n for n in (_as_node(obj) for obj in read_jsonl(path, schema=None)) if n]
    return nodes


def collect_chain(nodes: Iterable[KQANode]) -> Dict[str, List[KQANode]]:
    """Group nodes by paper_id and sort by step ascending."""
    by_pid: Dict[str, List[KQANode]] = {}
    for node in nodes:
        by_pid.setdefault(node.paper_id, []).append(node)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda n: n.step)
    return by_pid


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def compose_known(chain: List[KQANode], step: int, separator: str = "\n\n") -> Optional[str]:
    """Compose Known_i as K0 + concat(A_0..A_{i-1}).

    Returns None if step==0 and no K0 present or inputs are inconsistent.
    """
    if not chain:
        return None
    # find K0
    k0 = next((n.known for n in chain if n.step == 0), None)
    if k0 is None:
        return None
    if step == 0:
        return k0
    answers: List[str] = [n.answer for n in chain if 0 <= n.step < step]
    return separator.join([k0] + answers)


def verify_known_materialization(chain: List[KQANode], node: KQANode) -> Tuple[bool, Dict[str, Any]]:
    """Check that materialized `node.known` includes prior answers up to its prev_step.

    We don't expect exact string equality because LLM may reformat Known_i; instead,
    we verify that each A_{j} (j < node.step) appears (whitespace-insensitive) in node.known.
    """
    target = _normalize_space(node.known)
    prior = [n for n in chain if n.step < node.step]
    missing: List[int] = []
    for p in prior:
        if _normalize_space(p.answer) not in target:
            missing.append(p.step)
    return (len(missing) == 0, {"missing_answer_steps": missing, "checked": [p.step for p in prior]})


def head_tail_view(chain: List[KQANode], tail_step: int) -> Optional[Tuple[str, str, str]]:
    """Return (K0, Q_tail, A_tail) if available for a paper chain."""
    k0 = next((n.known for n in chain if n.step == 0), None)
    tail = next((n for n in chain if n.step == tail_step), None)
    if k0 is None or tail is None:
        return None
    return (k0, tail.question, tail.answer)
