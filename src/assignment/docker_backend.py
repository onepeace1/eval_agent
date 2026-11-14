"""Local Docker sandboxes: a drop-in replacement for the Modal ones.

`Environment` runs every agent command through a SWE-ReX server inside a
container. On Modal that container is a `modal.Sandbox`; with
``SANDBOX_BACKEND=docker`` (the default) it is a container on this machine,
started and stopped here, with the same image contents:

* `LocalImage` mirrors the small part of the `modal.Image` API the assignment
  uses (``from_dockerfile``, ``from_registry``, ``pip_install``,
  ``add_local_file``), so `build_testbed_image` and `ChessSandbox` describe an
  image the same way for either backend. Each step is built with ``docker
  build`` and tagged by a hash of its inputs, so an unchanged image is reused.
* `DockerSandboxDeployment` adds the SWE-ReX server to the image, runs it, and
  publishes forwarded ports on 127.0.0.1, where Modal would open tunnels.

Set ``SANDBOX_BACKEND=modal`` to go back to Modal.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from typing import Callable, Iterable

from swerex.deployment.docker import DockerDeployment
from swerex.runtime.config import RemoteRuntimeConfig
from swerex.runtime.remote import RemoteRuntime
from swerex.utils.free_port import find_free_port

logger = logging.getLogger(__name__)

BACKEND_VARIABLE = "SANDBOX_BACKEND"
BACKENDS = ("docker", "modal")

#: Every container and image this module creates carries this label, so a
#: leftover can be found with ``docker ps --filter label=assignment-sandbox``.
LABEL = "assignment-sandbox"
IMAGE_REPOSITORY = "assignment-sandbox"

#: Port the SWE-ReX server listens on inside the container. The same one the
#: Modal deployment uses, which keeps 8000 free for the chess server.
RUNTIME_PORT = 8880

#: The SWE-ReX server in the container matches the client driving it.
SWEREX_VERSION = importlib.metadata.version("swe-rex")
UV_IMAGE = "ghcr.io/astral-sh/uv:0.9.4"


def backend() -> str:
    """Which sandbox backend to use: ``docker`` unless ``SANDBOX_BACKEND`` says otherwise."""
    value = os.environ.get(BACKEND_VARIABLE, "docker").strip().lower() or "docker"
    if value not in BACKENDS:
        raise ValueError(f"{BACKEND_VARIABLE}={value!r}; expected one of {', '.join(BACKENDS)}")
    return value


def _docker(*args: str, capture: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=capture, text=True, **kwargs)


def image_exists(reference: str) -> bool:
    return _docker("image", "inspect", reference).returncode == 0


# -- Images -------------------------------------------------------------------


@dataclass(frozen=True)
class _Step:
    """One ``docker build`` on top of the previous step's image."""

    kind: str  # "registry" | "dockerfile" | "lines"
    reference: str = ""  # registry: the image name
    dockerfile: str = ""  # dockerfile: path to the Dockerfile
    context: str = ""  # dockerfile: build context directory
    ignore: Callable[[Path], bool] | None = field(default=None, compare=False)
    lines: tuple[str, ...] = ()  # lines: Dockerfile instructions after FROM
    files: tuple[tuple[str, str], ...] = ()  # lines: (local path, name in context)


@dataclass(frozen=True)
class LocalImage:
    """A Docker image built locally, described like a `modal.Image`.

    Nothing is built until `build` is called, normally by the deployment that
    runs it, so describing an image is free.
    """

    steps: tuple[_Step, ...]
    force_build: bool = False

    @classmethod
    def from_registry(cls, reference: str) -> "LocalImage":
        return cls((_Step("registry", reference=reference),))

    @classmethod
    def from_dockerfile(
        cls,
        path: str | PurePath,
        context_dir: str | PurePath | None = None,
        *,
        force_build: bool = False,
        ignore: Callable[[Path], bool] | None = None,
    ) -> "LocalImage":
        path = Path(path).resolve()
        context = Path(context_dir).resolve() if context_dir else path.parent
        step = _Step("dockerfile", dockerfile=str(path), context=str(context), ignore=ignore)
        return cls((step,), force_build=force_build)

    def _then(self, lines: Iterable[str], files: Iterable[tuple[str, str]] = ()) -> "LocalImage":
        return replace(self, steps=(*self.steps, _Step("lines", lines=tuple(lines), files=tuple(files))))

    def pip_install(self, *packages: str) -> "LocalImage":
        if not packages:
            return self
        return self._then([f"RUN python -m pip install --no-cache-dir {shlex.join(packages)}"])

    def add_local_file(self, local_path: str | PurePath, remote_path: str, copy: bool = True) -> "LocalImage":
        # `copy` exists for signature compatibility; a Docker layer always copies.
        local = Path(local_path).resolve()
        if not local.is_file():
            raise FileNotFoundError(local)
        name = f"file-{len(self.steps)}-{local.name}"
        return self._then([f"COPY {name} {remote_path}"], [(str(local), name)])

    def with_swerex(self) -> "LocalImage":
        """Add the SWE-ReX server, in its own Python, without touching the image's.

        uv fetches a standalone CPython into /opt/swerex, so this works on any
        glibc image whatever Python it ships (or none), and the testbed's own
        interpreter and packages stay exactly as the task built them.
        """
        return self._then([
            f"COPY --from={UV_IMAGE} /uv /opt/swerex/bin/uv",
            "RUN export UV_PYTHON_INSTALL_DIR=/opt/swerex/python UV_CACHE_DIR=/tmp/swerex-uv-cache"
            " && /opt/swerex/bin/uv venv --managed-python --python 3.12 /opt/swerex/venv"
            f" && /opt/swerex/bin/uv pip install --python /opt/swerex/venv/bin/python swe-rex=={SWEREX_VERSION}"
            " && rm -rf /tmp/swerex-uv-cache"
            " && ln -sf /opt/swerex/venv/bin/swerex-remote /usr/local/bin/swerex-remote",
        ])

    def build(self) -> str:
        """Build every step that is not already built, and return the final tag."""
        tag = ""
        for step in self.steps:
            tag = _build_step(step, tag, self.force_build)
        return tag


def _fingerprint(step: _Step, parent: str) -> str:
    digest = hashlib.sha256(f"{parent}\0{step.kind}\0{step.reference}\0".encode())
    for line in step.lines:
        digest.update(line.encode() + b"\0")
    for local, name in step.files:
        digest.update(name.encode() + b"\0" + Path(local).read_bytes())
    if step.kind == "dockerfile":
        digest.update(Path(step.dockerfile).read_bytes())
        for path in _context_files(step):
            digest.update(str(path).encode() + b"\0")
            digest.update(((Path(step.context) / path).read_bytes()))
    return digest.hexdigest()[:20]


def _context_files(step: _Step) -> list[Path]:
    root = Path(step.context)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and not (step.ignore and step.ignore(relative)):
            files.append(relative)
    return files


def _build_step(step: _Step, parent: str, force: bool) -> str:
    if step.kind == "registry":
        if not image_exists(step.reference):
            logger.info("Pulling %s", step.reference)
            pulled = _docker("pull", step.reference)
            if pulled.returncode != 0:
                raise RuntimeError(f"docker pull {step.reference} failed:\n{pulled.stderr.strip()}")
        return step.reference

    tag = f"{IMAGE_REPOSITORY}:{_fingerprint(step, parent)}"
    if not force and image_exists(tag):
        return tag

    with tempfile.TemporaryDirectory(prefix="assignment-build-") as scratch:
        context = Path(scratch) / "context"
        context.mkdir()
        if step.kind == "dockerfile":
            for relative in _context_files(step):
                target = context / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(step.context) / relative, target)
            dockerfile = Path(step.dockerfile).read_text()
        else:
            for local, name in step.files:
                shutil.copy2(local, context / name)
            dockerfile = "\n".join([f"FROM {parent}", *step.lines]) + "\n"
        dockerfile_path = Path(scratch) / "Dockerfile"
        dockerfile_path.write_text(dockerfile)

        command = ["build", "-t", tag, "--label", f"{LABEL}=1", "-f", str(dockerfile_path)]
        if force:
            command.append("--no-cache")
        logger.info("Building %s", tag)
        started = time.monotonic()
        built = _docker(*command, str(context))
        if built.returncode != 0:
            raise RuntimeError(f"docker build failed for {tag}:\n{(built.stdout + built.stderr)[-4000:]}")
        logger.info("Built %s in %.1fs", tag, time.monotonic() - started)
    return tag


def as_local_image(image) -> LocalImage:
    """Interpret anything `Environment` accepts as an image, for Docker."""
    if isinstance(image, LocalImage):
        return image
    if isinstance(image, PurePath) or (isinstance(image, str) and Path(image).is_file()):
        return LocalImage.from_dockerfile(image)
    if isinstance(image, str):
        return LocalImage.from_registry(image)
    raise TypeError(
        f"{type(image).__name__} cannot run with {BACKEND_VARIABLE}=docker. Pass an image name, "
        f"a Dockerfile path, or a LocalImage, or set {BACKEND_VARIABLE}=modal."
    )


# -- Deployment ---------------------------------------------------------------


class DockerSandboxDeployment(DockerDeployment):
    """A SWE-ReX deployment in a local container, shaped like the Modal one.

    The container lives at most ``deployment_timeout`` seconds, enforced inside
    it by ``timeout``, so one leaked by a crash still goes away on its own, as a
    Modal sandbox would. Ports in ``forwarded_ports`` are published on
    127.0.0.1; `url_for` is the counterpart of a Modal tunnel URL.
    """

    def __init__(
        self,
        *,
        image,
        startup_timeout: float = 600,
        runtime_timeout: float = 600,
        deployment_timeout: float = 600,
        forwarded_ports: Iterable[int] = (),
        docker_args: Iterable[str] = (),
    ):
        super().__init__(image="assignment-sandbox", startup_timeout=startup_timeout, pull="never")
        self._image = as_local_image(image)
        self._command_timeout = runtime_timeout
        self._deployment_timeout = deployment_timeout
        self._forwarded = list(dict.fromkeys(int(port) for port in forwarded_ports))
        self._extra_args = list(docker_args)
        self._host_ports: dict[int, int] = {}
        # How long one liveness probe may take while waiting for startup.
        self._runtime_timeout = 2.0

    def url_for(self, port: int) -> str:
        """The host URL for a forwarded container port."""
        try:
            return f"http://127.0.0.1:{self._host_ports[port]}"
        except KeyError as exc:
            forwarded = ", ".join(str(item) for item in sorted(self._host_ports)) or "none"
            raise ValueError(f"Port {port} was not forwarded. Available forwarded ports: {forwarded}.") from exc

    @property
    def running(self) -> bool:
        return self._container_process is not None and self._container_process.poll() is None

    async def start(self) -> None:
        if self._runtime is not None:
            self.logger.warning("Deployment is already started; ignoring duplicate start().")
            return

        self._hooks.on_custom_step("Building container image")
        image = self._image.with_swerex().build()
        self._config.image = image

        runtime_port = find_free_port()
        self._host_ports = {}
        publish = ["-p", f"127.0.0.1:{runtime_port}:{RUNTIME_PORT}"]
        for port in self._forwarded:
            host_port = find_free_port()
            while host_port == runtime_port or host_port in self._host_ports.values():
                host_port = find_free_port()
            self._host_ports[port] = host_port
            publish += ["-p", f"127.0.0.1:{host_port}:{port}"]

        token = self._get_token()
        name = f"assignment-sandbox-{uuid.uuid4().hex[:12]}"
        self._container_name = name
        command = [
            "docker", "run", "--rm", "--init",
            "--name", name,
            "--label", f"{LABEL}=1",
            *publish,
            *self._extra_args,
            image,
            "timeout", "-s", "KILL", str(int(self._deployment_timeout)),
            "swerex-remote", "--host", "0.0.0.0", "--port", str(RUNTIME_PORT), "--auth-token", token,
        ]
        self.logger.info("Starting container %s from %s", name, image)
        started = time.monotonic()
        self._container_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._hooks.on_custom_step("Starting runtime")
        self._runtime = RemoteRuntime.from_config(
            RemoteRuntimeConfig(
                host="http://127.0.0.1", port=runtime_port, timeout=self._command_timeout, auth_token=token
            )
        )
        try:
            await self._wait_until_alive(timeout=self._config.startup_timeout)
        except Exception as exc:
            await self.stop()
            raise RuntimeError(
                f"Container {name} started, but the SWE-ReX runtime did not become reachable: {exc}"
            ) from exc
        self.logger.info("Runtime ready in %.1fs", time.monotonic() - started)
