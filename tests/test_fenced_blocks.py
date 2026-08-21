import json

from infra.text.fenced_blocks import extract_preferred_fenced_block


def test_extract_fenced_json_ignores_backticks_inside_json_strings() -> None:
    # Regression test:
    # Many JSON payloads embed Markdown code blocks inside string fields (``` ... ```),
    # which used to confuse naive fence extraction (first "```" wins).
    text = """preface
```json
{
  "draft_question_explicit": "Pseudo-code:\\n```\\nresult = 0\\n```\\n",
  "x": 1
}
```
suffix
"""
    extracted = extract_preferred_fenced_block(text, preferred_langs=("json",))
    assert extracted is not None
    obj = json.loads(extracted)
    assert obj["x"] == 1


def test_extract_fenced_prefers_json_block_when_multiple_present() -> None:
    text = """```python
print("not json")
```
```json
{"ok": true}
```"""
    extracted = extract_preferred_fenced_block(text, preferred_langs=("json",))
    assert extracted is not None
    assert json.loads(extracted)["ok"] is True
