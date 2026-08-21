"""通用工具模块"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件，支持基础配置引用

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        from infra.config.config_loader import load_config_with_base, validate_inference_config

        # 使用增强的配置加载器
        config = load_config_with_base(config_path)

        # 验证推理服务配置
        if not validate_inference_config(config):
            logger.warning("推理服务配置验证失败，可能影响功能")

        return config

    except ImportError:
        # 回退到简单加载器
        logger.warning("使用简单配置加载器，不支持配置引用")
        return _load_config_simple(config_path)


def _load_config_simple(config_path: str) -> Dict[str, Any]:
    """简单配置加载器（回退方案）"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 环境变量替换
    config = _replace_env_vars(config)

    logger.info(f"已加载配置: {config_path}")
    return config


def _replace_env_vars(obj: Any) -> Any:
    """递归替换配置中的环境变量"""
    if isinstance(obj, dict):
        return {key: _replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        env_var = obj[2:-1]
        return os.getenv(env_var, obj)
    else:
        return obj


def ensure_dir(path: str) -> Path:
    """确保目录存在"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
