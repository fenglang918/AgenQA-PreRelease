"""配置加载器 - 支持基础配置引用和覆盖"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_config_with_base(config_path: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件，支持 base_config 引用和 overrides 覆盖

    Args:
        config_path: 主配置文件路径
        project_root: 项目根目录，用于解析相对路径

    Returns:
        合并后的配置字典
    """
    if project_root is None:
        project_root = os.getcwd()

    config_path = Path(project_root) / config_path
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 递归处理配置引用
    config = _resolve_config_references(config, project_root)

    # 环境变量替换
    config = _replace_env_vars(config)

    logger.info(f"已加载配置: {config_path}")
    return config


def _resolve_config_references(config: Any, project_root: str) -> Any:
    """递归解析配置引用"""
    if isinstance(config, dict):
        # 检查是否有 base_config 引用
        if 'base_config' in config and 'overrides' in config:
            base_config_path = config['base_config']
            overrides = config['overrides']

            # 加载基础配置
            base_path = Path(project_root) / base_config_path
            with open(base_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)

            # 递归解析基础配置中的引用
            base_config = _resolve_config_references(base_config, project_root)

            # 应用覆盖
            merged_config = _deep_merge(base_config, overrides)

            # 移除引用标记
            merged_config.pop('base_config', None)
            merged_config.pop('overrides', None)

            logger.debug(f"合并基础配置: {base_config_path}")
            return merged_config

        # 递归处理子字典
        return {key: _resolve_config_references(value, project_root)
                for key, value in config.items()}

    elif isinstance(config, list):
        # 递归处理列表
        return [_resolve_config_references(item, project_root) for item in config]

    else:
        # 基本类型直接返回
        return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典，override 中的值会覆盖 base 中的值"""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = _deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = value

    return result


def _replace_env_vars(obj: Any) -> Any:
    """递归替换配置中的环境变量"""
    if isinstance(obj, dict):
        return {key: _replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        # 支持两种形式：
        # - ${VAR}                      → 使用环境变量 VAR，若未设置则保留原字符串
        # - ${VAR:-default}            → 使用环境变量 VAR，若未设置则使用 default
        expr = obj[2:-1]
        env_name = expr
        default_val: Any = obj
        if ':-' in expr:
            parts = expr.split(':-', 1)
            env_name = parts[0]
            default_val = parts[1]
        return os.getenv(env_name, default_val)
    else:
        return obj


def _extract_generator_blocks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从配置中收集所有潜在的 generator 配置块。"""

    blocks: List[Dict[str, Any]] = []

    def _maybe_add(block: Any) -> None:
        if not isinstance(block, dict) or not block:
            return
        # Prefer explicit generator sub-block; otherwise allow the block itself
        # to serve as a generator-like config (service_id/service_type/api_base...).
        gen = block.get("generator")
        if isinstance(gen, dict) and gen:
            blocks.append(gen)
        else:
            if any(k in block for k in ("service_id", "model_name", "api_base", "base_url")):
                blocks.append(block)

    # 检查常见的顶层配置键
    for key in ['seed', 'extend_upgrade', 'solve', 'verify', 'init']:
        _maybe_add(config.get(key, {}) or {})

    # best-practice init：允许子模块覆盖
    init_block = config.get("init") or {}
    if isinstance(init_block, dict):
        _maybe_add((init_block.get("episode_seed") or {}) if isinstance(init_block.get("episode_seed"), dict) else {})
        _maybe_add((init_block.get("extract") or {}) if isinstance(init_block.get("extract"), dict) else {})

    # 新版 Agent 配置结构：director / final_commenter / operators / solvers
    _maybe_add(config.get("director") or {})
    _maybe_add(config.get("final_commenter") or {})

    ops = config.get("operators") or {}
    if isinstance(ops, dict):
        for _, op_block in ops.items():
            _maybe_add(op_block)

    solvers = config.get("solvers") or {}
    if isinstance(solvers, dict):
        for _, solver_block in solvers.items():
            _maybe_add(solver_block)

    # 检查 grow 块中的嵌套配置
    grow_block = config.get('grow', {}) or {}
    if isinstance(grow_block, dict):
        for item in grow_block.values():
            if isinstance(item, dict) and isinstance(item.get('generator'), dict):
                blocks.append(item['generator'])

    return blocks


def validate_inference_config(config: Dict[str, Any]) -> bool:
    """验证推理服务配置的完整性。

    检查各模块（seed、extend_upgrade、solve、verify、grow 等）的 generator。
    只要存在一个有效的生成器配置即视为通过。
    """

    generator_configs = _extract_generator_blocks(config)

    if not generator_configs:
        logger.error(
            "缺少推理服务 generator 配置（未在 seed/extend_upgrade/solve/verify/grow/director/operators/solvers 中找到）"
        )
        return False

    valid_found = False
    for generator_config in generator_configs:
        raw_service_type = generator_config.get('service_type') or 'private_endpoint'
        service_type = raw_service_type
        if service_type == 'local_service':
            logger.warning("service_type 'local_service' 已重命名为 'private_endpoint'，请更新配置文件")
            service_type = 'private_endpoint'

        if service_type != 'private_endpoint':
            logger.error(
                "当前版本仅支持通过 Service/llm_service 暴露的 OpenAI 兼容私有端点，请设置 service_type=private_endpoint"
            )
            continue

        # 如果提供了 service_id，则允许在运行时从 llm_service 自动解析 model_name 和 api_base
        service_id = generator_config.get('service_id')
        has_model_name = bool(generator_config.get('model_name'))
        has_api_base = bool(generator_config.get('api_base') or generator_config.get('base_url'))

        if not service_id and not has_model_name:
            logger.error("私有端点配置缺少 model_name（或 service_id 用于自动解析）")
            continue

        if not service_id and not has_api_base:
            logger.error("私有端点配置缺少 api_base/base_url（或 service_id 用于自动解析）")
            continue

        # 如果提供了 service_id，即使缺少某些字段也视为有效（会在运行时解析）
        if service_id:
            valid_found = True
        elif has_model_name and has_api_base:
            valid_found = True

    if valid_found:
        logger.info("推理服务配置验证通过")
    else:
        logger.error("未找到有效的推理服务 generator 配置")

    return valid_found
