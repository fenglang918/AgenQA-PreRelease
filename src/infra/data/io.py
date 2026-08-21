"""数据IO和模式校验模块"""

import json
import jsonschema
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


# JSON Schema 定义
PAPER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SciClone.Paper",
    "type": "object",
    "required": ["text"],
    "properties": {
        "text": {"type": "string"},
        "title": {"type": "string"},
        "abstract": {"type": "string"},
        "discipline": {"type": "string"},
        "meta": {"type": "object"}
    }
}

# 保留旧版模式供潜在兼容使用，但当前流程不会写入结构化 QA 结果
QA_RECORD_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SciClone.QARecord",
    "type": "object",
    "properties": {
        "paper_id": {"type": "string"},
        "discipline": {"type": "string"},
        "source": {"type": "string"},
        "keywords": {"type": "array"},
        "qas": {"type": "array"},
        "gen_params": {"type": "object"},
        "trace": {"type": "object"}
    }
}


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """验证数据是否符合模式"""
    try:
        jsonschema.validate(data, schema)
        return True
    except jsonschema.ValidationError as e:
        logger.warning(f"数据模式验证失败: {e}")
        return False


def read_jsonl(
    file_path: Union[str, Path],
    schema: Optional[Dict[str, Any]] = None,
    max_lines: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """读取JSONL文件

    Args:
        file_path: 文件路径
        schema: 可选的JSON模式验证
        max_lines: 最大读取行数

    Yields:
        解析后的JSON对象
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if schema and not validate_schema(data, schema):
                    logger.warning(f"跳过第{i+1}行，模式验证失败")
                    continue
                yield data
            except json.JSONDecodeError as e:
                logger.error(f"解析第{i+1}行JSON失败: {e}")
                continue


def write_jsonl(
    data_list: List[Dict[str, Any]],
    file_path: Union[str, Path],
    schema: Optional[Dict[str, Any]] = None,
    append: bool = False
) -> None:
    """写入JSONL文件

    Args:
        data_list: 数据列表
        file_path: 输出文件路径
        schema: 可选的JSON模式验证
        append: 是否追加写入
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as f:
        for data in data_list:
            if schema and not validate_schema(data, schema):
                logger.warning("跳过写入，数据模式验证失败")
                continue
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    logger.info(f"已写入 {len(data_list)} 条记录到 {file_path}")


def count_lines(file_path: Union[str, Path]) -> int:
    """统计文件行数"""
    file_path = Path(file_path)
    if not file_path.exists():
        return 0

    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def read_text_file(file_path: Union[str, Path]) -> str:
    """读取文本文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
