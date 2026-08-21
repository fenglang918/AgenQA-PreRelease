"""Domain seed walker: expand a domain into subdomains and sample leaf keywords.

Run:
  python -m infra.domain_seed_walk --api-base ... --model-name qwen3-max --root-domain "diffusion models"
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from utils import ensure_dir, load_config

logger = logging.getLogger("domain_seed_walk")


@dataclass
class WalkConfig:
    root_domain: str
    depth: int = 4
    branching: int = 5
    keywords_per_leaf: int = 10
    lang: str = "en"  # "en" | "zh"
    output_dir: Path | None = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    generator: Dict[str, Any] = None  # type: ignore[assignment]


def _get_by_path(obj: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    cur: Any = obj
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
    return cur if isinstance(cur, dict) else None


def _ensure_generation_overrides(generator: Dict[str, Any], temperature: Optional[float], max_tokens: Optional[int]) -> None:
    if temperature is None and max_tokens is None:
        return
    gen = generator.get("generation")
    if not isinstance(gen, dict):
        gen = {}
        generator["generation"] = gen
    if temperature is not None:
        gen["temperature"] = float(temperature)
    if max_tokens is not None:
        gen["max_tokens"] = int(max_tokens)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content or "", encoding="utf-8")


def _run_llm(
    session,
    chat_args: Dict[str, Any],
    prompt: str,
    *,
    snapshot_dir: Path,
    name_prefix: str,
    lang: str,
) -> str:
    ensure_dir(str(snapshot_dir))
    _write_text(snapshot_dir / f"{name_prefix}.prompt.txt", prompt)
    messages = build_messages_with_background(prompt, lang=lang or "en")
    try:
        resp = session.chat(messages, **chat_args)
    except Exception as exc:  # noqa: BLE001
        _write_text(snapshot_dir / f"{name_prefix}.error.txt", f"{type(exc).__name__}: {exc}")
        raise
    try:
        _write_text(snapshot_dir / f"{name_prefix}.response.json", json.dumps(resp, ensure_ascii=False, indent=2))
    except Exception:
        _write_text(snapshot_dir / f"{name_prefix}.response.txt", str(resp))
    text = session.extract_text(resp, default="")
    _write_text(snapshot_dir / f"{name_prefix}.extracted.txt", text)
    return text or ""


def _expand_prompt(domain: str, branching: int, *, lang: str) -> str:
    lang_norm = (lang or "en").strip().lower() or "en"
    if lang_norm in {"zh", "cn", "zh-cn", "zh-hans"}:
        return (
            "给定一个研究领域（domain），请将其分解为更具体的子领域（subdomains）。\n\n"
            f"Domain: {domain}\n\n"
            f"只输出 JSON，且必须包含恰好 {branching} 个子领域：\n"
            "{\n"
            '  "subdomains": [\n'
            '    {"name": "...", "context_tags": ["...", "..."]}\n'
            "  ]\n"
            "}\n\n"
            "规则：\n"
            f"- 必须输出恰好 {branching} 个条目。\n"
            "- `name`/`context_tags` 的内容用中文（键名保持英文，不要翻译键名）。\n"
            "- 每个 item.name 是具体研究子方向（2–10 个中文词/短语为宜）。\n"
            "- 每个 item.context_tags 给出 2–4 个短标签，用于钉死技术语境/消歧。\n"
            "- 避免泛化标题（如“简介/应用/概述/未来工作”）。\n"
            "- 若术语有歧义，必须使用 root domain + path context 消歧，且保持同一技术语境，不得跨领域漂移。\n"
            "- 不要编号，不要多余解释文本。"
        )
    return (
        "You are given a research domain. Decompose it into subdomains.\n\n"
        f"Domain: {domain}\n\n"
        f"Return JSON ONLY with exactly {branching} unique subdomains:\n"
        "{\n"
        '  "subdomains": [\n'
        '    {"name": "...", "context_tags": ["...", "..."]}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        f"- Output exactly {branching} items.\n"
        "- English only (keep JSON keys in English).\n"
        "- Each item.name should be a concrete research subtopic (2–8 words).\n"
        "- Each item.context_tags should have 2–4 short tags that pin down the intended technical context.\n"
        "- Avoid generic headings like 'introduction' or 'applications'.\n"
        "- Do not change the meaning of ambiguous terms; resolve ambiguity using the root domain + path context.\n"
        "- Do not drift across fields: each subdomain must be a strict specialization within the same technical context.\n"
        "- No numbering, no extra text."
    )

def _render_path_context(path_trace: List[Dict[str, Any]]) -> str:
    if not path_trace:
        return ""
    lines: List[str] = []
    lines.append("Path context (chosen trajectory so far):")
    for e in path_trace:
        if not isinstance(e, dict):
            continue
        lvl = e.get("level")
        inp = e.get("input_domain")
        ch = e.get("chosen")
        if inp and ch:
            tags = e.get("chosen_context_tags")
            if isinstance(tags, list):
                tags = [str(x).strip() for x in tags if str(x).strip()]
            else:
                tags = []
            tags_part = f" [tags: {', '.join(tags)}]" if tags else ""
            lines.append(f"- Level {lvl}: {inp} -> {ch}{tags_part}")
    return "\n".join(lines).strip()


def _expand_prompt_with_context(
    root_domain: str,
    domain: str,
    branching: int,
    path_trace: List[Dict[str, Any]],
    *,
    lang: str,
) -> str:
    base = _expand_prompt(domain, branching, lang=lang)
    ctx = _render_path_context(path_trace)
    lang_norm = (lang or "en").strip().lower() or "en"
    root_line = (
        f"Root domain (intended context): {root_domain}".strip()
        if lang_norm not in {"zh", "cn", "zh-cn", "zh-hans"}
        else f"Root domain（全局语境约束）: {root_domain}".strip()
    )
    if not ctx:
        if lang_norm in {"zh", "cn", "zh-cn", "zh-hans"}:
            return base.replace(f"Domain: {domain}", f"{root_line}\n\nDomain: {domain}")
        return (
            base.replace("You are given a research domain.", "You are given a research domain and a root domain context.")
            .replace(f"Domain: {domain}", f"{root_line}\n\nDomain: {domain}")
        )
    # Put context before output schema to anchor the model, but keep requirements identical.
    parts = base.split("\n\n", 2)
    if len(parts) >= 2:
        if lang_norm in {"zh", "cn", "zh-cn", "zh-hans"}:
            head = parts[0]
        else:
            head = parts[0].replace("You are given a research domain.", "You are given a research domain and a root domain context.")
        return head + "\n\n" + root_line + "\n\n" + ctx + "\n\n" + "\n\n".join(parts[1:])
    return base + "\n\n" + ctx


def _keywords_prompt(root_domain: str, leaf_domain: str, k: int, path_trace: List[Dict[str, Any]]) -> str:
    ctx = _render_path_context(path_trace)
    return (
        "Generate problem keywords for the given leaf domain, staying consistent with the root domain and the path context.\n\n"
        f"Root domain: {root_domain}\n\n"
        f"{ctx}\n\n"
        f"Leaf domain: {leaf_domain}\n\n"
        f"Return JSON ONLY with exactly {k} keywords:\n"
        "{\n"
        '  "problem_keywords": ["..."]\n'
        "}\n\n"
        "Rules:\n"
        f"- Output exactly {k} items.\n"
        "- English only.\n"
        "- Use specific technical phrases (2–6 words).\n"
        "- Keywords must stay within the same technical context as the root/path/leaf.\n"
        "- Prefer keywords that help construct non-trivial problems (e.g., constraints, failure modes, trade-offs, metrics), not just a glossary.\n"
        "- No numbering, no extra text."
    )


def _keywords_prompt_lang(root_domain: str, leaf_domain: str, k: int, path_trace: List[Dict[str, Any]], *, lang: str) -> str:
    lang_norm = (lang or "en").strip().lower() or "en"
    ctx = _render_path_context(path_trace)
    if lang_norm in {"zh", "cn", "zh-cn", "zh-hans"}:
        return (
            "请为给定的叶子 domain 生成用于出题的 problem keywords，并保持与 root domain 与 path context 的技术语境一致。\n\n"
            f"Root domain: {root_domain}\n\n"
            f"{ctx}\n\n"
            f"Leaf domain: {leaf_domain}\n\n"
            f"只输出 JSON，且必须包含恰好 {k} 个关键词：\n"
            "{\n"
            '  "problem_keywords": ["..."]\n'
            "}\n\n"
            "规则：\n"
            f"- 必须输出恰好 {k} 个条目。\n"
            "- 关键词内容用中文（键名保持英文）。\n"
            "- 用具体技术短语（2–10 个中文词/短语为宜）。\n"
            "- 关键词必须与 root/path/leaf 的同一技术语境一致，不得跨领域漂移。\n"
            "- 优先包含有助于出“非平庸题”的要素（约束、失败模式、权衡、指标等），而不是纯术语表。\n"
            "- 不要编号，不要多余解释文本。"
        )
    return _keywords_prompt(root_domain, leaf_domain, k, path_trace)


def _extract_first_json_value(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        start = m.start()
        try:
            val, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(val, dict):
            return val
        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], dict):
            return val[0]
        return None
    return None


def _iter_json_candidates(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    candidates: List[str] = []
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        inner = (m.group(1) or "").strip()
        if inner:
            candidates.append(inner)
    candidates.append(raw)
    return candidates


def _cleanup_list_items(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    for it in items:
        s = str(it or "").strip()
        if not s:
            continue
        s = s.strip(" \t\r\n,")
        s = s.strip("`")
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        s = s.strip('"\', \t\r\n')
        if not s:
            continue
        cleaned.append(s)

    seen: set[str] = set()
    uniq: List[str] = []
    for s in cleaned:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def _salvage_string_list(raw_text: str, key: str) -> Optional[List[str]]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    m = re.search(rf"\"{re.escape(key)}\"\s*:\s*\[(.*?)\]", raw_text, flags=re.IGNORECASE | re.DOTALL)
    inner = None
    if m:
        inner = m.group(1)
    else:
        m2 = re.search(r"\[(.*?)\]", raw_text, flags=re.DOTALL)
        if m2:
            inner = m2.group(1)
    if inner is None:
        return None

    parts: List[str] = []
    for line in inner.splitlines():
        line = line.strip()
        if not line:
            continue
        parts.extend([p for p in line.split(",") if p.strip()])
    items = _cleanup_list_items(parts)
    return items or None


def _extract_bracket_payload(raw_text: str, key: str) -> Optional[str]:
    """Extract the raw content inside the first [...] after a JSON key.

    Returns the inside substring without the surrounding brackets, or None.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    m = re.search(rf"\"{re.escape(key)}\"\s*:\s*\[(.*?)\]", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    m2 = re.search(r"\[(.*?)\]", raw_text, flags=re.DOTALL)
    if m2:
        return m2.group(1)
    return None


def _salvage_subdomain_objects(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """Best-effort salvage for subdomains objects, tolerant of minor JSON issues.

    Handles patterns like:
      {"name": Strong consistency protocols, "context_tags": ["a","b"]}
    where name might be missing quotes.
    """
    inner = _extract_bracket_payload(raw_text, "subdomains")
    if inner is None:
        return None
    objs: List[Dict[str, Any]] = []

    # Split into object-ish chunks.
    for m in re.finditer(r"\{(.*?)\}", inner, flags=re.DOTALL):
        block = m.group(1) or ""
        # name: either "..." or unquoted token until comma/newline/}
        m_name = re.search(r"\"name\"\s*:\s*(\"(.*?)\"|([^,\n\r}]+))", block, flags=re.IGNORECASE | re.DOTALL)
        if not m_name:
            continue
        name = m_name.group(2) if m_name.group(2) is not None else (m_name.group(3) or "")
        name = str(name).strip().strip('"\',')
        if not name:
            continue
        # context_tags: grab quoted strings inside [...]
        tags: List[str] = []
        m_tags = re.search(r"\"context_tags\"\s*:\s*\[(.*?)\]", block, flags=re.IGNORECASE | re.DOTALL)
        if m_tags:
            raw_tags = m_tags.group(1) or ""
            tags = [t.strip() for t in re.findall(r"\"([^\"]+)\"", raw_tags) if t.strip()]
        objs.append({"name": name, "context_tags": tags})

    objs = _normalize_subdomains(objs)
    return objs or None


def _normalize_subdomains(items: Any, *, fallback_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if fallback_tags is None:
        fallback_tags = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict):
            name = it.get("name") or it.get("subdomain") or it.get("title") or ""
            name = str(name).strip()
            raw_tags = it.get("context_tags") or it.get("tags") or []
            tags: List[str] = []
            if isinstance(raw_tags, str) and raw_tags.strip():
                tags = [x.strip() for x in raw_tags.split(",") if x.strip()]
            elif isinstance(raw_tags, list):
                tags = [str(x).strip() for x in raw_tags if str(x).strip()]
            if name:
                out.append({"name": name, "context_tags": tags})
            continue
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.append({"name": s, "context_tags": list(fallback_tags)})
            continue
        s = str(it).strip()
        if s:
            out.append({"name": s, "context_tags": list(fallback_tags)})
    # Dedup by name (case-insensitive)
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for obj in out:
        key = str(obj.get("name") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(obj)
    return uniq


def _coerce_string_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    # Dedup
    seen: set[str] = set()
    uniq: List[str] = []
    for s in cleaned:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def _clean_json_obj(text: str, required_keys: List[str]) -> Dict[str, Any]:
    for cand in _iter_json_candidates(text):
        obj = _extract_first_json_value(cand)
        if not isinstance(obj, dict):
            continue
        keys_lower = {str(k).lower() for k in obj.keys()}
        if all(str(k).lower() in keys_lower for k in required_keys):
            return obj
    raise ValueError("failed to extract required JSON object")


def _parse_json(
    raw_text: str,
    *,
    required_keys: List[str],
    task_name: str,
    snapshot_dir: Path,
) -> Dict[str, Any]:
    try:
        obj = _clean_json_obj(raw_text or "", required_keys=required_keys)
        _write_text(snapshot_dir / f"{task_name}.cleaned.json", json.dumps(obj, ensure_ascii=False, indent=2))
        return obj
    except Exception:
        pass

    if required_keys == ["subdomains"]:
        objs = _salvage_subdomain_objects(raw_text or "")
        if objs:
            obj = {"subdomains": objs}
            _write_text(snapshot_dir / f"{task_name}.cleaned.json", json.dumps(obj, ensure_ascii=False, indent=2))
            return obj
        items = _salvage_string_list(raw_text or "", "subdomains")
        if items:
            obj = {"subdomains": [{"name": s, "context_tags": []} for s in items if s and s != "{"]}
            _write_text(snapshot_dir / f"{task_name}.cleaned.json", json.dumps(obj, ensure_ascii=False, indent=2))
            return obj
    if required_keys == ["problem_keywords"]:
        items = _salvage_string_list(raw_text or "", "problem_keywords")
        if items:
            obj = {"problem_keywords": items}
            _write_text(snapshot_dir / f"{task_name}.cleaned.json", json.dumps(obj, ensure_ascii=False, indent=2))
            return obj

    raise ValueError("failed to extract required JSON object")


def run_walk(conf: WalkConfig) -> Dict[str, Any]:
    resolved = resolve_inference(conf.generator)
    _ensure_generation_overrides(conf.generator, conf.temperature, conf.max_tokens)
    session = resolved.session
    chat_args = dict(resolved.chat_args)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = conf.output_dir or Path("data/domain_seed_walk") / f"run_{now}"
    ensure_dir(str(out_dir))

    path_trace: List[Dict[str, Any]] = []
    current = conf.root_domain.strip()
    if not current:
        raise ValueError("root_domain must be non-empty")

    for level in range(1, conf.depth + 1):
        level_dir = Path(out_dir) / f"level_{level:02d}"
        prompt = (
            _expand_prompt(current, conf.branching, lang=conf.lang)
            if getattr(conf, "no_trace_context", False)
            else _expand_prompt_with_context(conf.root_domain, current, conf.branching, path_trace, lang=conf.lang)
        )
        raw = _run_llm(session, chat_args, prompt, snapshot_dir=level_dir, name_prefix="expand", lang=conf.lang)
        parsed = _parse_json(raw, required_keys=["subdomains"], task_name="expand", snapshot_dir=level_dir)

        subs = parsed.get("subdomains")
        subs_norm = _normalize_subdomains(subs)
        subs = [x for x in subs_norm if isinstance(x, dict) and str(x.get("name") or "").strip()]
        if not subs:
            raise ValueError("expand: empty subdomains")
        if len(subs) > conf.branching:
            subs = subs[: conf.branching]

        chosen_obj = random.choice(subs)
        chosen = str(chosen_obj.get("name") or "").strip()
        chosen_tags = chosen_obj.get("context_tags") if isinstance(chosen_obj.get("context_tags"), list) else []
        chosen_tags = [str(x).strip() for x in chosen_tags if str(x).strip()]
        path_trace.append({"level": level, "input_domain": current, "candidates": subs, "chosen": chosen})
        path_trace[-1]["chosen_context_tags"] = chosen_tags
        current = chosen

    leaf_domain = current
    leaf_dir = Path(out_dir) / "leaf"
    leaf_prompt = _keywords_prompt_lang(conf.root_domain, leaf_domain, conf.keywords_per_leaf, path_trace, lang=conf.lang)
    leaf_raw = _run_llm(session, chat_args, leaf_prompt, snapshot_dir=leaf_dir, name_prefix="keywords", lang=conf.lang)
    leaf_parsed = _parse_json(leaf_raw, required_keys=["problem_keywords"], task_name="keywords", snapshot_dir=leaf_dir)

    keywords = leaf_parsed.get("problem_keywords")
    keywords = _coerce_string_list(keywords)
    if len(keywords) > conf.keywords_per_leaf:
        keywords = keywords[: conf.keywords_per_leaf]

    result = {
        "schema_version": 2,
        "root_domain": conf.root_domain,
        "depth": conf.depth,
        "branching": conf.branching,
        "lang": conf.lang,
        "leaf_domain": leaf_domain,
        "problem_keywords": keywords,
        "path_trace": path_trace,
        "provenance": resolved.provenance,
        "run_dir": str(out_dir),
        "timestamp": now,
    }
    _write_text(Path(out_dir) / "result.json", json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Domain seed walk (single-path, random).")
    p.add_argument("--config", default="config/agent_idealab.yaml", help="YAML config path (ignored if --api-base is set)")
    p.add_argument("--generator-path", default="init.generator", help="dot path to generator block in config")
    p.add_argument(
        "--api-base",
        help="Direct mode: OpenAI-compatible base URL (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1). "
        "If set, --config/--generator-path are ignored.",
    )
    p.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="Direct mode: environment variable name holding API key")
    p.add_argument("--model-name", default="qwen3-max", help="Direct mode: model name")
    p.add_argument("--lang", choices=["en", "zh"], default="en", help="output language for names/tags/keywords (default: en)")
    p.add_argument("--root-domain", required=True, help="root domain (string)")
    p.add_argument("--depth", type=int, default=4, help="max depth to walk")
    p.add_argument("--branching", type=int, default=5, help="branching factor per level (LLM output size)")
    p.add_argument("--keywords-per-leaf", type=int, default=10, help="number of leaf keywords")
    p.add_argument("--output-dir", help="output directory (default: data/domain_seed_walk/run_YYYYMMDD_HHMMSS)")
    p.add_argument("--no-playback", action="store_true", help="do not write <run_dir>/playback.md")
    p.add_argument(
        "--no-trace-context",
        action="store_true",
        help="do not include the full chosen path trace in each expand prompt",
    )
    p.add_argument("--temperature", type=float, help="override generation.temperature")
    p.add_argument("--max-tokens", type=int, help="override generation.max_tokens")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_arg_parser().parse_args()

    generator: Dict[str, Any]
    if isinstance(args.api_base, str) and args.api_base.strip():
        api_key_env = str(args.api_key_env or "").strip() or "DASHSCOPE_API_KEY"
        generator = {
            "service_type": "private_endpoint",
            "api_base": args.api_base.strip(),
            "api_key": f"${api_key_env}",
            "model_name": str(args.model_name or "").strip() or "qwen3-max",
            "generation": {},
            "client": {"stream": False},
        }
        logger.info(
            "Direct generator mode: api_base=%s model=%s api_key_env=%s",
            generator["api_base"],
            generator["model_name"],
            api_key_env,
        )
    else:
        config = load_config(args.config)
        generator = _get_by_path(config, args.generator_path) or {}
        if not isinstance(generator, dict) or not generator:
            raise ValueError(f"generator not found at path: {args.generator_path}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    conf = WalkConfig(
        root_domain=args.root_domain,
        depth=max(1, int(args.depth)),
        branching=max(1, int(args.branching)),
        keywords_per_leaf=max(1, int(args.keywords_per_leaf)),
        lang=str(args.lang or "en").strip().lower() or "en",
        output_dir=output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        generator=generator,
    )
    setattr(conf, "no_trace_context", bool(getattr(args, "no_trace_context", False)))
    result = run_walk(conf)
    if not bool(getattr(args, "no_playback", False)):
        try:
            from .playback import build_playback

            run_dir = Path(result["run_dir"])
            playback_md = run_dir / "playback.md"
            playback_md.write_text(build_playback(run_dir, preview_chars=0), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write playback.md (ignored).")
    logger.info("Done. result.json at %s", result["run_dir"])


if __name__ == "__main__":
    main()
