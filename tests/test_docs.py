import contextlib
import importlib.metadata
import io
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import gymnasium
import pytest

import raycer
from raycer.cli import main

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
DOC_PAGES = ["api", "scenarios", "observations", "recording", "reproducibility", "benchmark"]
HEADINGS = [
    "Installation",
    "Quickstart",
    "Scenarios",
    "Observations",
    "Actions",
    "Reward",
    "CLI",
    "For ViZDoom users",
    "Citing and license",
]
SUBCOMMANDS = ("benchmark", "render-sample", "list", "play")


def _doc_files() -> list[Path]:
    """Return the README and every docs page."""
    return [README, *sorted(DOCS.glob("*.md"))]


def _test_blocks() -> list[str]:
    """Return the README python blocks directly preceded by a <!-- test --> line."""
    pattern = r"^<!-- test -->\n```python\n(.*?)^```$"
    return re.findall(pattern, README.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)


def _help(argv: list[str]) -> str:
    """Return the --help output of the CLI for argv."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exc:
        main([*argv, "--help"])
    assert exc.value.code == 0
    return out.getvalue()


def test_readme_code_blocks_run_DOC04(tmp_path: Path) -> None:
    blocks = _test_blocks()
    assert len(blocks) >= 2
    assert sum(1 for b in blocks for line in b.splitlines() if line.strip()) <= 15
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
    for block in blocks:
        result = subprocess.run([sys.executable, "-c", block], cwd=tmp_path, env=env)
        assert result.returncode == 0, block


def test_gym_ids_registered_DOC04() -> None:
    readme = README.read_text(encoding="utf-8")
    ids = {
        m
        for path in _doc_files()
        for m in re.findall(r"Raycer/[A-Za-z]+(?:-Continuous)?-v\d+", path.read_text("utf-8"))
    }
    assert "Raycer/Oval-v0" in ids
    assert ids <= set(gymnasium.registry), ids - set(gymnasium.registry)
    for name in raycer.list_scenarios():
        assert name in readme


def test_cli_flags_exist_DOC04() -> None:
    helps = {sub: _help([sub]) for sub in SUBCOMMANDS}
    top = _help([])
    checked = dict.fromkeys(SUBCOMMANDS, 0)
    for path in _doc_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            flags = re.findall(r"(?<![\w-])--[a-z][a-z-]*", line)
            subs = re.findall(rf"\braycer ({'|'.join(SUBCOMMANDS)})\b", line)
            for sub in subs:
                for flag in flags:
                    assert re.search(rf"{flag}(?![\w-])", helps[sub]), f"{path.name}: {sub} {flag}"
                    checked[sub] += 1
            for flag in re.findall(r"\braycer (--[a-z][a-z-]*)", line):
                assert re.search(rf"{flag}(?![\w-])", top), f"{path.name}: raycer {flag}"
    assert all(checked[sub] for sub in ("benchmark", "render-sample", "play")), checked


def test_readme_order_DOC01() -> None:
    text = README.read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
    assert headings[: len(HEADINGS)] == HEADINGS
    intro = text.split("\n## Installation\n", 1)[0]
    assert "](docs/img/buffers.png)" in intro
    sentence = [ln for ln in intro.splitlines()[1:] if ln.strip() and not ln.startswith("![")]
    assert sentence


def test_buffers_image_DOC01() -> None:
    data = (DOCS / "img" / "buffers.png").read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) <= 102_400


def test_docs_pages_DOC02() -> None:
    for page in DOC_PAGES:
        assert (DOCS / f"{page}.md").read_text(encoding="utf-8").startswith("# ")
    observations = (DOCS / "observations.md").read_text(encoding="utf-8")
    for name in [*raycer.Game().get_label_names().values(), *raycer.GAME_VARIABLES]:
        assert name in observations
    reproducibility = (DOCS / "reproducibility.md").read_text(encoding="utf-8")
    for split in ("train", "val", "test"):
        assert split in reproducibility


def test_citation_and_changelog_DOC04() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines()
    for line in [
        "cff-version: 1.2.0",
        'title: "Raycer: A Raycasting Racing Environment for Machine Learning Research"',
        "authors:",
        "license: MIT",
        f"version: {raycer.__version__}",
        "repository-code: https://github.com/JulianErt/Raycer",
    ]:
        assert line in citation
    assert any(ln.startswith("message: ") for ln in citation)
    assert any("JulianErt" in ln for ln in citation)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.startswith("# Changelog\n")
    assert "https://keepachangelog.com/en/1.1.0/" in changelog
    assert "Semantic Versioning" in changelog
    assert re.search(r"^## \[Unreleased\]$", changelog, re.MULTILINE)
    assert re.search(r"^## \[0\.1\.0\] - \d{4}-\d{2}-\d{2}$", changelog, re.MULTILINE)
    assert "### Added" in changelog.split("## [0.1.0]", 1)[1]


def test_version_and_extras_PKG04() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert version == importlib.metadata.version("raycer") == raycer.__version__
    pygame_reqs = [r for r in importlib.metadata.requires("raycer") or [] if "pygame" in r]
    assert pygame_reqs
    for req in pygame_reqs:
        assert re.search(r"""extra\s*==\s*["']human["']""", req), req
