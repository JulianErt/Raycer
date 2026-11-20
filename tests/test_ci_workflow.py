"""QA-40: the GitHub Actions workflow parses and covers every required item."""

import json
from pathlib import Path
from typing import Any

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


class YamlSubsetError(ValueError):
    """Raised for anything outside the YAML subset the workflow may use."""


def _strip_comment(text: str) -> str:
    """Drop a trailing ` # comment` from unquoted text."""
    pos = text.find(" #")
    return text[:pos].rstrip() if pos >= 0 else text


def _quoted(text: str) -> tuple[str, str]:
    """Parse a leading quoted scalar; return its value and the remaining text."""
    if text[0] == '"':
        end = 1
        while end < len(text) and text[end] != '"':
            end += 2 if text[end] == "\\" else 1
        if end >= len(text):
            raise YamlSubsetError(f"unterminated string: {text}")
        return str(json.loads(text[: end + 1])), text[end + 1 :]
    end = 1
    while True:
        end = text.find("'", end)
        if end < 0:
            raise YamlSubsetError(f"unterminated string: {text}")
        if text[end + 1 : end + 2] != "'":
            return text[1:end].replace("''", "'"), text[end + 1 :]
        end += 2


def _scalar(text: str) -> str:
    """Parse a plain or quoted scalar (values stay strings, no type coercion)."""
    if text and text[0] in "\"'":
        value, rest = _quoted(text)
        if _strip_comment(rest).strip():
            raise YamlSubsetError(f"text after string: {text}")
        return value
    if text[:1] in set("&*!>|{}[]%@`,") or text.startswith("---"):
        raise YamlSubsetError(f"unsupported syntax: {text}")
    return _strip_comment(text).strip()


def _value(text: str) -> str | list[str]:
    """Parse an inline value: a flow sequence of scalars or a single scalar."""
    if not text.startswith("["):
        return _scalar(text)
    rest = text[1:].strip()
    items: list[str] = []
    while not rest.startswith("]"):
        if rest and rest[0] in "\"'":
            item, rest = _quoted(rest)
        else:
            end = min((i for i in (rest.find(","), rest.find("]")) if i >= 0), default=-1)
            if end < 0:
                raise YamlSubsetError(f"unterminated flow sequence: {text}")
            item, rest = _scalar(rest[:end]), rest[end:]
        items.append(item)
        rest = rest.strip()
        if rest.startswith(","):
            rest = rest[1:].strip()
        elif not rest.startswith("]"):
            raise YamlSubsetError(f"bad flow sequence: {text}")
    if _strip_comment(rest[1:]).strip():
        raise YamlSubsetError(f"text after flow sequence: {text}")
    return items


class _Parser:
    def __init__(self, text: str) -> None:
        self.raw = text.splitlines()
        self.lines: list[tuple[int, int, str]] = []  # (raw line number, indent, content)
        for number, line in enumerate(self.raw):
            content = line.lstrip(" ")
            if "\t" in line[: len(line) - len(content) + 1]:
                raise YamlSubsetError(f"tab in indentation on line {number + 1}")
            if content and not content.startswith("#"):
                self.lines.append((number, len(line) - len(content), content.rstrip()))
        self.pos = 0

    def parse(self) -> Any:
        if not self.lines:
            raise YamlSubsetError("empty document")
        value = self.block(self.lines[0][1])
        if self.pos < len(self.lines):
            raise YamlSubsetError(f"unexpected indentation on line {self.lines[self.pos][0] + 1}")
        return value

    def nested(self, indent: int) -> Any:
        """Parse the block under a key or dash, which must be indented further."""
        if self.pos >= len(self.lines) or self.lines[self.pos][1] <= indent:
            raise YamlSubsetError("missing nested block")
        return self.block(self.lines[self.pos][1])

    def block(self, indent: int) -> Any:
        content = self.lines[self.pos][2]
        if content == "-" or content.startswith("- "):
            return self.sequence(indent)
        return self.mapping(indent)

    def sequence(self, indent: int) -> list[Any]:
        items: list[Any] = []
        while self.pos < len(self.lines) and self.lines[self.pos][1] == indent:
            number, _, content = self.lines[self.pos]
            if not (content == "-" or content.startswith("- ")):
                raise YamlSubsetError(f"expected '- ' on line {number + 1}")
            rest = content[1:].lstrip(" ")
            if not rest:
                self.pos += 1
                items.append(self.nested(indent))
            elif ":" in _strip_comment(rest) and rest[:1] not in "\"'[":
                # An inline mapping item: re-read the rest as a mapping one column deeper.
                inner = indent + len(content) - len(rest)
                self.lines[self.pos] = (number, inner, rest)
                items.append(self.mapping(inner))
            else:
                self.pos += 1
                items.append(_value(rest))
        return items

    def mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.pos < len(self.lines) and self.lines[self.pos][1] == indent:
            number, _, content = self.lines[self.pos]
            key, sep, rest = content.partition(": ")
            if not sep:
                if not content.endswith(":"):
                    raise YamlSubsetError(f"expected 'key: value' on line {number + 1}")
                key, rest = content[:-1], ""
            key = _scalar(key)
            if not key or key in result:
                raise YamlSubsetError(f"empty or duplicate key on line {number + 1}")
            rest = rest.strip()
            self.pos += 1
            if rest == "|":
                result[key] = self.literal(number, indent)
            elif _strip_comment(rest) and not rest.startswith("#"):
                result[key] = _value(rest)
            elif self.pos < len(self.lines) and self.lines[self.pos][1] > indent:
                result[key] = self.nested(indent)
            else:
                result[key] = ""
        if self.pos < len(self.lines) and self.lines[self.pos][1] > indent:
            raise YamlSubsetError(f"inconsistent indentation on line {self.lines[self.pos][0] + 1}")
        return result

    def literal(self, number: int, indent: int) -> str:
        """Read a `|` block scalar from the raw lines following line `number`."""
        while self.pos < len(self.lines) and self.lines[self.pos][1] > indent:
            self.pos += 1
        end = self.lines[self.pos][0] if self.pos < len(self.lines) else len(self.raw)
        body = self.raw[number + 1 : end]
        while body and not body[-1].strip():
            body.pop()
        if not body:
            raise YamlSubsetError(f"empty literal block on line {number + 1}")
        width = len(body[0]) - len(body[0].lstrip(" "))
        if width <= indent or any(line.strip() and line[:width].strip() for line in body):
            raise YamlSubsetError(f"bad literal block indentation after line {number + 1}")
        return "\n".join(line[width:] for line in body) + "\n"


def parse_yaml(text: str) -> Any:
    """Parse the small YAML subset used by the CI workflow."""
    return _Parser(text).parse()


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    result = parse_yaml(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    return result


def _run_steps(job: dict[str, Any]) -> list[str]:
    return [step.get("run", "") for step in job["steps"]]


def test_workflow_parses() -> None:
    result = parse_yaml(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    assert {"on", "jobs"} <= result.keys()


def test_triggers(workflow: dict[str, Any]) -> None:
    on = workflow["on"]
    assert on["push"]["branches"] == ["main"]
    assert "pull_request" in on
    assert "workflow_dispatch" in on


def test_matrix_os(workflow: dict[str, Any]) -> None:
    check = workflow["jobs"]["check"]
    assert check["runs-on"] == "${{ matrix.os }}"
    assert set(check["strategy"]["matrix"]["os"]) == {"ubuntu-latest", "macos-latest", "windows-latest"}


def test_python_version(workflow: dict[str, Any]) -> None:
    check = workflow["jobs"]["check"]
    assert check["strategy"]["matrix"]["python-version"] == ["3.12"]
    setup = next(s for s in check["steps"] if s.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"


def test_setup_uv_cache(workflow: dict[str, Any]) -> None:
    for job in workflow["jobs"].values():
        uses = [s for s in job["steps"] if s.get("uses", "").startswith("astral-sh/setup-uv@v")]
        assert len(uses) == 1
        assert uses[0]["with"]["enable-cache"] == "true"


def test_gate_step_with_min_sps(workflow: dict[str, Any]) -> None:
    steps = workflow["jobs"]["check"]["steps"]
    assert steps[0]["uses"] == "actions/checkout@v4"
    gate = next(s for s in steps if "bash scripts/check.sh" in s.get("run", ""))
    assert gate["env"]["RAYCER_MIN_SPS"] == "100"


def test_build_after_gate(workflow: dict[str, Any]) -> None:
    runs = _run_steps(workflow["jobs"]["check"])
    gate = next(i for i, run in enumerate(runs) if "bash scripts/check.sh" in run)
    assert any("uv build" in run for run in runs[gate + 1 :])


def test_wheel_smoke_install(workflow: dict[str, Any]) -> None:
    runs = _run_steps(workflow["jobs"]["check"])
    build = next(i for i, run in enumerate(runs) if "uv build" in run)
    smoke = [run for run in runs[build + 1 :] if "uv venv" in run]
    assert len(smoke) == 1
    assert "uv pip install --python" in smoke[0]
    assert "dist/*.whl" in smoke[0]
    assert "import raycer" in smoke[0]
    assert "--no-project" in smoke[0]
    assert "raycer --version" in smoke[0]


def test_check_timeout_and_fail_fast(workflow: dict[str, Any]) -> None:
    check = workflow["jobs"]["check"]
    assert check["timeout-minutes"] == "15"
    assert check["strategy"]["fail-fast"] == "false"


def test_slow_job_manual(workflow: dict[str, Any]) -> None:
    slow = workflow["jobs"]["slow"]
    assert slow["if"] == "github.event_name == 'workflow_dispatch'"
    assert slow["timeout-minutes"]
    assert any("pytest -m slow" in run for run in _run_steps(slow))


def test_parser_reads_subset() -> None:
    text = "a: [x, \"y, z\", 'q']  # c\nb:\n  - k: |\n      one\n\n      two\n    m: 1\n  - plain\n"
    assert parse_yaml(text) == {
        "a": ["x", "y, z", "q"],
        "b": [{"k": "one\n\ntwo\n", "m": "1"}, "plain"],
    }


@pytest.mark.parametrize(
    "text",
    [
        "a:\n  b: 1\n   c: 2\n",
        "a: &anchor 1\n",
        "a: {b: 1}\n",
        "a: >\n  folded\n",
        "a: 1\na: 2\n",
        "just text\n",
        'a: "open\n',
    ],
)
def test_parser_rejects_unsupported(text: str) -> None:
    with pytest.raises(YamlSubsetError):
        parse_yaml(text)


def test_parser_rejects_tab_indent() -> None:
    text = WORKFLOW.read_text(encoding="utf-8").replace("\n  ", "\n\t", 1)
    with pytest.raises(YamlSubsetError):
        parse_yaml(text)
