import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden", action="store_true", default=False, help="rewrite tests/golden/*.npz"
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    """Whether golden files should be rewritten instead of compared."""
    return bool(request.config.getoption("--update-golden"))
