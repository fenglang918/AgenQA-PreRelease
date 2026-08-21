"""结构化日志模块"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """结构化JSON日志格式化器"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加额外字段
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """用于终端输出的彩色日志格式化器（不影响文件日志）。"""

    COLORS = {
        logging.DEBUG: "\033[37m",   # 灰
        logging.INFO: "\033[36m",    # 青
        logging.WARNING: "\033[33m", # 黄
        logging.ERROR: "\033[31m",   # 红
        logging.CRITICAL: "\033[41m",  # 红底
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        if not color:
            return msg
        return f"{color}{msg}{self.RESET}"


def setup_logger(
    name: str,
    config: Dict[str, Any],
    stage: Optional[str] = None
) -> logging.Logger:
    """设置结构化日志记录器

    Args:
        name: 记录器名称
        config: 日志配置
        stage: 当前阶段（如 seed, generator 等）

    Returns:
        配置好的日志记录器
    """
    desired_level = getattr(logging, config.get("level", "INFO"))
    root_logger = logging.getLogger()
    if root_logger.level > desired_level:
        root_logger.setLevel(desired_level)

    log_file = config.get("file_path", "logs/agenqa.log")
    if stage:
        log_file = f"logs/{stage}.jsonl"

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 确保结构化文件处理器只注册一次
    log_path = Path(log_file).resolve()
    has_file_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_filename = getattr(handler, "baseFilename", None)
            if base_filename and Path(base_filename).resolve() == log_path:
                has_file_handler = True
                break

    if not has_file_handler:
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # 终端输出：仅在尚未添加 StreamHandler 时注册一个彩色控制台 handler
    has_console = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    if not has_console:
        console_handler = logging.StreamHandler()
        console_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        console_handler.setFormatter(ColorFormatter(console_fmt, datefmt="%H:%M:%S"))
        console_handler.setLevel(desired_level)
        root_logger.addHandler(console_handler)

    logger = logging.getLogger(name)
    logger.propagate = True
    return logger


def log_with_extra(logger: logging.Logger, level: str, message: str, **extra_fields):
    """记录带额外字段的日志"""
    record = logging.LogRecord(
        name=logger.name,
        level=getattr(logging, level.upper()),
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None
    )
    record.extra_fields = extra_fields
    logger.handle(record)
