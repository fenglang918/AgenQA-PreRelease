"""ExecutableDiagnose role prompt (Python style)."""

from __future__ import annotations

from textwrap import dedent

from agenqa.domain.diagnose_schema import diagnose_output_schema_text

__all__ = [
    "EXECUTABLE_DIAGNOSE_V1",
    "EXECUTABLE_DIAGNOSE_V1_EN",
]


EXECUTABLE_DIAGNOSE_V1 = dedent(
    f"""\
    # ExecutableDiagnose（诊断 executable step 的失败原因与修复方向）

    ## Input
    - step: $step
    - director_notes: $director_notes
    - executable_tail_json: $executable_tail_json
    - eval_error: $eval_error

    ## Task
    你是 Diagnose 角色：只负责识别问题与给出修复建议，不直接重写题目。
    基于 executable_tail 的接口/描述/依赖与执行错误，判断失败的主因与最小修复方向：
    - 协议/契约问题：function_header/return_line 与实现不一致、签名不可调用、返回类型不稳定；
    - 依赖问题：不在白名单、缺失 import、环境不确定；
    - 可测性/确定性问题：隐式随机、I/O、副作用、数值不稳定；
    - 可解性/可扩展性问题：跳过前序接口、不满足单步递进、职责边界混乱。

    输出应当可执行（给出明确的修复动作），但不要直接输出修复后的代码。

    ## Output format
    仅输出一个 JSON 对象，必须用 ```json 代码块包裹，顶层字段严格为：
    {diagnose_output_schema_text()}
    """
)


EXECUTABLE_DIAGNOSE_V1_EN = dedent(
    f"""\
    # ExecutableDiagnose (diagnose failures in a executable step)

    ## Input
    - step: $step
    - director_notes: $director_notes
    - executable_tail_json: $executable_tail_json
    - eval_error: $eval_error

    ## Task
    You are the Diagnose role: identify issues and propose repair directions; do NOT rewrite the step/code.
    Based on the step spec (interface/description/dependencies) and the execution error, diagnose the root cause and propose minimal, actionable fixes:
    - contract issues: signature/return mismatch, non-callable interface, unstable return type;
    - dependency issues: out-of-whitelist deps, missing imports, environment dependence;
    - testability/determinism issues: randomness, I/O, side effects, numeric instability;
    - progression issues: bypassing previous step, violating single-step progression, unclear responsibility boundary.

    ## Output format
    Output one JSON object only, wrapped in ```json. Top-level fields:
    {diagnose_output_schema_text()}
    """
)
