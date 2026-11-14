"""The chess server in a local Docker sandbox: test_chess_sandbox.py without Modal."""

import subprocess

import pytest

from assignment.chess_sandbox import ChessSandbox, IllegalMove
from assignment.docker_backend import LABEL

pytestmark = pytest.mark.docker


def running_sandboxes() -> set[str]:
    listed = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"label={LABEL}"],
        capture_output=True, text=True, check=True,
    )
    return set(listed.stdout.split())


def test_chess_app_is_playable_through_local_port(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    before = running_sandboxes()

    with ChessSandbox() as sandbox:
        assert len(running_sandboxes() - before) == 1
        assert sandbox.server_url.startswith("http://127.0.0.1:")

        current = sandbox.state()
        assert current["turn"] == "white"
        assert current["history"] == []

        with pytest.raises(IllegalMove):
            sandbox.play("e2e5")

        # The files run_python needs are in the image, beside the testbed.
        listed = sandbox.execute("ls /opt/assignment")["output"].split()
        assert {"chess_server.py", "chess_tools.py", "sandbox_python.py"} <= set(listed)

        assert sandbox.reset()["history"] == []

    assert not running_sandboxes() - before, "Chess sandbox was not removed after the test."
