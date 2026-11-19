import pytest

import raycer
from raycer.cli import main


def test_version_matches_metadata() -> None:
    assert raycer.__version__ == "0.2.0"


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert raycer.__version__ in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[]])
def test_cli_exits_zero(argv: list[str]) -> None:
    assert main(argv) == 0
