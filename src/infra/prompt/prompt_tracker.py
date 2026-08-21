"""Prompt 使用跟踪与归档工具。

提供统一的日志输出与结果目录内的 prompt 快照写入，便于复现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

from utils import ensure_dir


def log_using_prompt(logger: logging.Logger, prompt_path: Path) -> None:
    """在日志中记录所用 Prompt 路径。"""
    try:
        logger.info("Using prompt: %s", str(prompt_path))
    except Exception:
        # 避免日志异常影响主流程
        pass


def snapshot_prompt_used(
    prompt_path: Path,
    output_dir: Path,
    *,
    content: Optional[str] = None,
    name_prefix: str = "prompt_used.",
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """在结果目录保存一份当次使用的 Prompt 文本。

    Args:
        prompt_path: 源 Prompt 文件路径
        output_dir: 结果输出目录
        content: 若已加载的模板文本（建议传入，保证与当次一致）；未提供则从文件读取
        name_prefix: 目标文件名前缀。为避免快照文件名冗余，若前缀以
            "prompt_used." 开头，会在落盘时自动剥离该公共前缀（调用点无需改动）。
        logger: 用于记录告警（可选）

    Returns:
        写入成功的目标路径，失败返回 None。
    """
    try:
        ensure_dir(str(output_dir))
        clean_prefix = (name_prefix or "").strip()
        if clean_prefix.startswith("prompt_used."):
            clean_prefix = clean_prefix[len("prompt_used.") :]
        elif clean_prefix == "prompt_used":
            clean_prefix = ""
        clean_prefix = clean_prefix.lstrip(".")
        dest = Path(output_dir) / f"{clean_prefix}{Path(prompt_path).name}"
        if dest.exists():
            return dest
        text = content if content is not None else Path(prompt_path).read_text(encoding="utf-8")
        Path(dest).write_text(text, encoding="utf-8")
        return dest
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("写入 prompt 备份失败: %s", exc)
        return None


def snapshot_rendered_prompt(
    prompt_text: str,
    output_dir: Path,
    *,
    filename: str = "prompt_rendered.txt",
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """将渲染后的 prompt 文本写入指定目录，便于事后查看。

    不依赖模板路径，只记录实际发送给模型的文本。
    """
    try:
        ensure_dir(str(output_dir))
        dest = Path(output_dir) / filename
        dest.write_text(prompt_text, encoding="utf-8")
        return dest
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("写入渲染后的 prompt 失败: %s", exc)
        return None
