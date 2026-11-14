"""The sandbox environment on the local Docker backend.

The Docker counterpart of test_env.py: real containers, so slower than the unit
tests but free. Run with `pytest -m docker` (or `make test-docker`).
"""

import subprocess

import pytest

from assignment.docker_backend import LABEL
from assignment.env import Environment

pytestmark = pytest.mark.docker


def running_sandboxes() -> list[str]:
    listed = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"label={LABEL}"],
        capture_output=True, text=True, check=True,
    )
    return listed.stdout.split()


@pytest.fixture(scope="module")
def env(monkeypatch_module):
    monkeypatch_module.setenv("SANDBOX_BACKEND", "docker")
    before = set(running_sandboxes())
    env = Environment()
    assert len(set(running_sandboxes()) - before) == 1, "expected exactly one new sandbox container"
    yield env
    env.stop()
    assert not set(running_sandboxes()) - before, "sandbox container still running after stop()"


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def test_shell_command(env):
    output = env.execute("echo 'hello, world'")
    assert output["returncode"] == 0
    assert output["output"] == "hello, world\n"


def test_argv_command(env):
    output = env.execute(["echo", "hello, world"], shell=False)
    assert output["returncode"] == 0
    assert output["output"] == "hello, world\n"


def test_failing_command(env):
    output = env.execute("python -c 'print(\"hello, world\"); raise Exception(\"test exception\")'")
    assert output["returncode"] != 0
    assert "hello, world\n" in output["output"]
    assert "test exception" in output["output"]


def test_testbed_python_is_untouched(env):
    """The SWE-ReX server brings its own Python; `python` is still the image's."""
    output = env.execute("command -v python; python -c 'import swerex' 2>&1 | tail -1")
    assert output["output"].splitlines()[0] == "/usr/local/bin/python"
    assert "ModuleNotFoundError" in output["output"]


def test_reports_linux_platform(env):
    assert env.system == "Linux"
    assert env.is_alive()
