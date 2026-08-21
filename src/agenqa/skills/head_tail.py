"""Head–Tail 组合器

从一条多步迭代链中抽掉中间过程，输出仅包含头部已知(K_0)与尾部题问/答案(Q_n/A_n)的 K/Q/A 记录，
可直接供 `solve` 管道消费，形成理论上需要多步推理才能解答的题目（例如 k0,q5,a5）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import logging
import re

from infra.data.io import read_jsonl, write_jsonl
from agenqa.domain.known_tree import KnownTree
from agenqa.domain.known_utils import parse_known_to_dict
from utils import ensure_dir


logger = logging.getLogger(__name__)


@dataclass
class HeadTailConfig:
    """运行配置（纯离线合并，无需推理会话）。"""

    # 预留扩展位，当前无需配置项
    pass


class HeadTailComposer:
    """基于 step=0 和 step=n 的 KQA 记录，输出 head–tail（k0,qn,an）。"""

    def __init__(self, config: Optional[HeadTailConfig] = None) -> None:
        self.config = config or HeadTailConfig()

    def run(
        self,
        output: Path,
        run_dir: Optional[Path] = None,
        head_kqa: Optional[Path] = None,
        tail_kqa: Optional[Path] = None,
        head_step: int = 0,
        tail_step: Optional[int] = None,
        append: bool = False,
    ) -> Path:
        """组合并写出 head–tail KQA。

        Args:
            output: 输出文件路径（.jsonl 或目录）。
            run_dir: 运行目录（包含 legacy init/extend-upgrade 的产物）；当未显式给出 head_kqa/tail_kqa 时启用自动探测。
            head_kqa: 明确指定 head(step=head_step) 的 KQA 路径（优先级高于 run_dir 自动探测）。
            tail_kqa: 明确指定 tail(step=tail_step) 的 KQA 路径（优先级高于 run_dir 自动探测）。
            head_step: 头部 step，默认 0。
            tail_step: 尾部 step，未指定时将从 run_dir 中自动取最大 step。
            append: 是否以追加模式写出。

        Returns:
            最终输出文件路径。
        """

        # 解析输入文件
        resolved_head, resolved_tail, resolved_tail_step = self._resolve_inputs(
            run_dir, head_kqa, tail_kqa, head_step, tail_step
        )

        # 读取 head/tail 记录并按 paper_id 对齐
        head_map = self._load_head(resolved_head, expect_step=head_step)
        tail_map = self._load_tail(resolved_tail, expect_step=resolved_tail_step)

        # 组合输出
        composed: List[Dict[str, object]] = []
        missing = 0
        matched = 0
        for pid, payload in head_map.items():
            if isinstance(payload, tuple) and len(payload) == 2:
                known, subj_head = payload
            else:
                known, subj_head = payload, None
            tail = tail_map.get(pid)
            if not tail:
                missing += 1
                continue
            matched += 1
            if isinstance(tail, tuple) and len(tail) >= 2:
                q, a = tail[0], tail[1]
                subj_tail = tail[2] if len(tail) >= 3 else None
            else:
                q, a, subj_tail = tail, None, None
            subject = subj_head or subj_tail
            # Trim known to head-tail view (v2). Fall back to legacy known_0 layout when needed.
            parsed_known = known
            if isinstance(parsed_known, str):
                parsed_known = parse_known_to_dict(parsed_known) or parsed_known
            if isinstance(parsed_known, dict) and (
                "episode_seed" in parsed_known or "premise_bank" in parsed_known
            ):
                trimmed_known = KnownTree.build_path_solver_view(parsed_known, resolved_tail_step)
            elif isinstance(parsed_known, dict) and "known_0" in parsed_known:
                trimmed_known = {"known_0": parsed_known.get("known_0", ""), "history": []}
            elif isinstance(parsed_known, str):
                trimmed_known = {"known_0": parsed_known, "history": []}
            else:
                trimmed_known = {"known_0": str(parsed_known), "history": []}
            composed.append(
                {
                    "paper_id": pid,
                    "step_head": head_step,
                    "step_tail": resolved_tail_step,
                    "known": trimmed_known,
                    "question": q,
                    "answer": a,
                    "subject": subject,
                    "source": "head_tail",
                    "chain": f"k{head_step},q{resolved_tail_step},a{resolved_tail_step}",
                }
            )

        if not composed:
            raise RuntimeError(
                "未能组合出任何 head–tail 记录，请检查输入是否对齐或 step 选择是否正确"
            )

        # 规范化输出路径
        out_path = self._resolve_output_path(output, resolved_tail_step)
        ensure_dir(str(out_path.parent))
        write_jsonl(composed, out_path, schema=None, append=append)

        logger.info(
            "Head–Tail 组合完成：matched=%d, missing=%d, output=%s",
            matched,
            missing,
            out_path,
        )
        return out_path

    # ---------------- internal helpers ----------------

    def _resolve_inputs(
        self,
        run_dir: Optional[Path],
        head_kqa: Optional[Path],
        tail_kqa: Optional[Path],
        head_step: int,
        tail_step: Optional[int],
    ) -> Tuple[Path, Path, int]:
        if head_kqa and tail_kqa:
            resolved_tail_step = self._infer_tail_step_from_name_or_content(tail_kqa, tail_step)
            return Path(head_kqa), Path(tail_kqa), resolved_tail_step

        if not run_dir:
            raise ValueError("未提供 run_dir，且未显式指定 head_kqa/tail_kqa")

        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"运行目录不存在: {run_dir}")

        # 寻找 head（优先 qa_init *_kqa.jsonl，回退到任意 step=head_step 的 KQA 文件）
        head = head_kqa or self._auto_find_head_kqa(run_dir, head_step)
        # 寻找 tail（优先指定 step，否则取 extend_kqa_step_{max}.jsonl）
        tail, resolved_tail_step = self._auto_find_tail_kqa(run_dir, tail_step)
        return head, tail, resolved_tail_step

    def _auto_find_head_kqa(self, run_dir: Path, head_step: int) -> Path:
        # 典型文件名：qa_init_raw_step_0_kqa.jsonl
        candidate = next(
            (p for p in run_dir.glob("*qa_init*_kqa.jsonl") if p.is_file()),
            None,
        )
        if candidate:
            return candidate

        # 回退：任何记录中 step==head_step 的 KQA 文件
        for p in run_dir.glob("*_kqa*.jsonl"):
            try:
                first = next(read_jsonl(p, schema=None, max_lines=1), None)
                if isinstance(first, dict) and int(first.get("step", -1)) == head_step:
                    return p
            except Exception:
                continue
        raise FileNotFoundError(f"未能在 {run_dir} 找到 head(step={head_step}) 的 KQA 文件")

    def _auto_find_tail_kqa(self, run_dir: Path, tail_step: Optional[int]) -> Tuple[Path, int]:
        pattern = re.compile(r"extend_kqa_step_(\d+)\.jsonl$")
        files: List[Tuple[int, Path]] = []
        for p in run_dir.glob("extend_kqa_step_*.jsonl"):
            m = pattern.search(p.name)
            if m:
                try:
                    s = int(m.group(1))
                    files.append((s, p))
                except ValueError:
                    continue
        if not files:
            raise FileNotFoundError(f"未在 {run_dir} 找到任何 extend_kqa_step_*.jsonl 作为 tail")

        if tail_step is None:
            s, tail = max(files, key=lambda t: t[0])
            return tail, s

        # 指定了 tail_step：精确匹配
        for s, p in files:
            if s == tail_step:
                return p, s
        raise FileNotFoundError(f"未找到 tail_step={tail_step} 的文件 extend_kqa_step_{tail_step}.jsonl")

    def _infer_tail_step_from_name_or_content(self, path: Path, hint: Optional[int]) -> int:
        if hint is not None:
            return int(hint)
        m = re.search(r"extend_kqa_step_(\d+)\.jsonl$", path.name)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        # 回退：读首条记录的 step
        try:
            first = next(read_jsonl(path, schema=None, max_lines=1), None)
            if isinstance(first, dict) and "step" in first:
                return int(first["step"])  # type: ignore[arg-type]
        except Exception:
            pass
        raise ValueError(f"无法从 {path} 推断 tail step，请显式提供 --tail-step")

    def _load_head(self, path: Path, expect_step: int) -> Dict[str, Tuple[object, Optional[str]]]:
        """读取 head KQA：返回 {paper_id -> (known, subject)}。

        known 可为 str 或树形 dict；subject 可为空。
        """
        head_map: Dict[str, Tuple[object, Optional[str]]] = {}
        for rec in self._iter_kqa_records(path):
            if not isinstance(rec, dict):
                continue
            pid = rec.get("paper_id")
            known = rec.get("known") or rec.get("Known")
            subj = rec.get("subject") if isinstance(rec.get("subject"), str) else None
            step = rec.get("step")
            if pid and (isinstance(known, str) or isinstance(known, dict)):
                if step is not None and int(step) != expect_step:
                    # 允许存在，但不作为 head（多源文件场景）
                    continue
                head_map[str(pid)] = (known, subj)
        if not head_map:
            raise RuntimeError(f"head 文件无有效记录或不含 step={expect_step}: {path}")
        return head_map

    def _load_tail(self, path: Path, expect_step: int) -> Dict[str, Tuple[str, str, Optional[str]]]:
        """读取 tail KQA：返回 {paper_id -> (question_n, answer_n, subject)}。subject 可为空。"""
        tail_map: Dict[str, Tuple[str, str, Optional[str]]] = {}
        for rec in self._iter_kqa_records(path):
            if not isinstance(rec, dict):
                continue
            pid = rec.get("paper_id")
            q = rec.get("question")
            a = rec.get("answer") or rec.get("Answer")
            subj = rec.get("subject") if isinstance(rec.get("subject"), str) else None
            step = rec.get("step")
            if step is not None and int(step) != expect_step:
                continue
            if pid and isinstance(q, str) and isinstance(a, str):
                tail_map[str(pid)] = (q, a, subj)
        if not tail_map:
            raise RuntimeError(f"tail 文件无有效记录或不含 step={expect_step}: {path}")
        return tail_map

    def _resolve_output_path(self, output: Path, tail_step: int) -> Path:
        output = Path(output)
        if output.suffix:
            return output
        # 目录：生成默认文件名
        out_dir = ensure_dir(str(output))
        return Path(out_dir) / f"head_tail_kqa_k0_q{tail_step}.jsonl"

    def _iter_kqa_records(self, path: Path) -> Iterable[Dict[str, object]]:
        """迭代 KQA 记录，兼容 JSONL 与连续 JSON 对象两种形式。"""
        text: str
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception:
            # 读不了就回落 JSONL 迭代
            yield from read_jsonl(path, schema=None)
            return

        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
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
                    yield obj
                idx = end
            except json.JSONDecodeError:
                yielded = 0
                break

        if yielded == 0:
            # 回落：逐行 JSONL
            yield from read_jsonl(path, schema=None)
