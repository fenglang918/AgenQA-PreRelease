# tests/

全量回归测试，覆盖 infra 工具层与 agenqa pipeline 的核心逻辑。均为纯单元/集成测试，不依赖真实 LLM 调用（通过 mock 或固定 fixture）。

## 运行

```bash
pytest tests/ -x -q          # 快速跑，遇错即停
pytest tests/ -v              # 详细输出
pytest tests/test_fenced_blocks.py  # 单文件
```

## 文件索引

### infra/text
| 文件 | 覆盖点 |
|---|---|
| `test_fenced_blocks.py` | markdown fenced block 解析边界 |
| `test_json_sanitize.py` | JSON 修复与清洗策略 |

### infra/llm
| 文件 | 覆盖点 |
|---|---|
| `test_service_client_http_error.py` | HTTP 错误码处理与重试逻辑 |
| `test_llm_probe_openai_compat_api.py` | OpenAI 兼容接口冒烟 |

### Director / 题型路由
| 文件 | 覆盖点 |
|---|---|
| `test_director_question_type_allowed_list.py` | Director 对 allowed_question_types 的路由过滤 |
| `test_question_type_policy_allowed_for_step.py` | 每步题型准入策略 |
| `test_symbolic_only_question_types.py` | `--symbolic-only` 模式下题型约束 |
| `test_symbolic_constraints_per_qtype.py` | 各题型的符号约束规则 |

### Solve / 判分
| 文件 | 覆盖点 |
|---|---|
| `test_solve_multi_strong_path_default.py` | Multi-Strong solver 默认路径 |
| `test_solve_apply_expression_judge_mode.py` | expression judge 模式激活条件 |
| `test_solve_expression_judge_mcq_fastpath.py` | MCQ fast-path 判分 |
| `test_numeric_judge_strict.py` | Numeric 严格判分边界 |
| `test_solver_tool_policy.py` | solver tool-use 策略 |

### Consensus
| 文件 | 覆盖点 |
|---|---|
| `test_consensus_defaults.py` | consensus 默认行为（none/always 与 alias 校验） |
| `test_consensus_detect_status_tool.py` | consensus 状态检测工具 |
| `test_solver_consensus_use_locked_question_type.py` | consensus 复用 locked question type |

### Revise
| 文件 | 覆盖点 |
|---|---|
| `test_revise_mode_world_contract.py` | world_contract revise 触发与合并 |
| `test_revise_mode_answer_contract.py` | answer_contract revise 触发 |
| `test_revise_mode_step_dir.py` | revise 落盘目录命名规则 |
| `test_revise_inherits_locked_question_type.py` | revise 继承 locked question type |

### Contract / 约束
| 文件 | 覆盖点 |
|---|---|
| `test_world_contract_object.py` | WorldContract 对象构建与字段校验 |
| `test_world_contract_merge.py` | 多步 world contract 合并逻辑 |
| `test_answer_contract_bank_lite.py` | ACB-lite bank 构建与查重 |

### CLI / 配置覆盖
| 文件 | 覆盖点 |
|---|---|
| `test_cli_override_allowed_question_types.py` | CLI `--allowed-question-types` 生效 |
| `test_cli_override_idealab_main_model.py` | CLI `--main-model` 覆盖 idealab 默认值 |

### 其他
| 文件 | 覆盖点 |
|---|---|
| `test_known_utils_format.py` | KQA 记录格式工具 |
| `test_python_executor_temp_dir_fallback.py` | code_verifier 临时目录 fallback |
| `conftest.py` | 全局 fixture（mock LLM、临时目录等） |
