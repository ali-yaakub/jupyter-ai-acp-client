"""
Windows compatibility for spawning and terminating ACP agent subprocesses.

Windows breaks the ACP client in four independent ways. Fixing any one of them
alone still leaves the personas unusable, so they are all handled here:

1. `jupyter_server` downgrades the event loop to `SelectorEventLoop` on Windows
   (`ServerApp._init_asyncio_patch`), and that loop cannot create subprocesses:
   `asyncio.create_subprocess_exec` raises `NotImplementedError`.

2. npm installs console scripts as `.cmd` shims, but `CreateProcessW` only
   appends `.exe` when resolving a bare name. Spawning `claude-agent-acp`
   raises `FileNotFoundError` even though the command is on `PATH` and
   `shutil.which` finds it.

3. Windows has no process groups in the POSIX sense, so `os.killpg`,
   `os.getpgid` and `signal.SIGKILL` do not exist at all — referencing them
   raises `AttributeError` rather than something the existing `OSError`
   handlers catch.

4. A `.cmd` shim runs under an intermediate `cmd.exe`. Killing that leaves the
   real agent (a `node` grandchild) orphaned, so termination has to kill the
   whole tree.

`create_subprocess()` and `terminate_process()` are no-ops on POSIX beyond
delegating to the stdlib, so callers do not need to branch on platform.
"""

from __future__ import annotations

import asyncio
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from asyncio.subprocess import Process
from typing import Any, Iterable, Optional, Union

IS_WINDOWS = sys.platform == "win32"

# Matches the `limit` passed to `asyncio.create_subprocess_exec` elsewhere in
# this package. ACP frames can be large (notebook attachments, file diffs).
DEFAULT_STREAM_LIMIT = 50 * 1024 * 1024

# Keep the console window hidden when spawning `.cmd` shims, which would
# otherwise flash a terminal on every agent start.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

AnyProcess = Union[Process, "WindowsProcess"]


def resolve_executable(argv: Iterable[str]) -> list[str]:
    """
    Resolve `argv[0]` to a full path on Windows so `.cmd`/`.bat` shims are
    found.

    `CreateProcessW` only tries the `.exe` extension when given a bare name,
    so `claude-agent-acp` fails even though npm placed `claude-agent-acp.cmd`
    on `PATH`. `shutil.which` honours `PATHEXT` and finds the shim.

    Returns `argv` unchanged on POSIX, when it is already a path, or when the
    command cannot be found — in the last case the caller still gets a
    `FileNotFoundError` naming the command the user actually configured.
    """
    args = list(argv)
    if not IS_WINDOWS or not args or not args[0]:
        return args

    command = args[0]
    # An explicit path is the caller's choice; don't second-guess it.
    if os.path.dirname(command):
        return args

    resolved = shutil.which(command)
    if resolved:
        args[0] = resolved
    return args


async def create_subprocess(*argv: str, **kwargs: Any) -> AnyProcess:
    """
    Spawn a subprocess, working on any event loop and on any platform.

    Resolves `.cmd` shims on Windows, then tries the native asyncio path first
    so POSIX — and Windows servers configured to keep the Proactor loop — get
    real `asyncio.subprocess.Process` objects with no threads involved. Only
    when the running loop cannot spawn subprocesses at all does it fall back to
    the `subprocess.Popen` bridge below.
    """
    args = resolve_executable(argv)
    try:
        return await asyncio.create_subprocess_exec(*args, **kwargs)
    except NotImplementedError:
        if not IS_WINDOWS:
            raise
        return _create_subprocess_via_popen(args, **kwargs)


def _create_subprocess_via_popen(
    args: list[str],
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    limit: int = DEFAULT_STREAM_LIMIT,
    env: Optional[dict[str, str]] = None,
    cwd: Any = None,
    # `start_new_session` is silently ignored by CPython on Windows (the
    # parameter is literally named `unused_start_new_session` there), so it is
    # accepted and dropped rather than passed through to imply a process group
    # that does not exist. Tree termination is handled by `terminate_process`.
    start_new_session: bool = False,
    **_ignored: Any,
) -> "WindowsProcess":
    """
    Spawn via `subprocess.Popen` and bridge its pipes onto the running loop.

    `bufsize=0` is load-bearing: it makes `Popen.stdout` a raw `FileIO` whose
    `read(n)` returns as soon as any bytes arrive. With the default buffering
    it is a `BufferedReader` that blocks until `n` bytes or EOF, which
    deadlocks against ACP's newline-delimited request/response protocol.
    """
    # `asyncio.subprocess.PIPE`/`DEVNULL`/`STDOUT` are the very same sentinels
    # as their `subprocess` counterparts, so the redirections pass through
    # unchanged. Translating them would quietly turn a requested DEVNULL into
    # an open pipe, leaving a command that reads stdin waiting forever.
    popen_kwargs: dict[str, Any] = {
        "stdin": stdin,
        "stdout": stdout,
        "stderr": stderr,
        "bufsize": 0,
        "creationflags": _NO_WINDOW,
    }
    if env is not None:
        popen_kwargs["env"] = env
    if cwd is not None:
        popen_kwargs["cwd"] = cwd

    popen = subprocess.Popen(args, **popen_kwargs)
    return WindowsProcess(popen, asyncio.get_running_loop(), limit=limit)


class _WriteTransport(asyncio.WriteTransport):
    """
    Write side of the bridge, backed by a dedicated thread.

    Writes are queued rather than performed inline. A blocking write on a full
    pipe would otherwise stall the entire Jupyter server event loop, which is
    reachable in practice by sending an agent a large notebook attachment.
    """

    def __init__(self, pipe: Any) -> None:
        super().__init__()
        self._pipe = pipe
        self._queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._closing = False
        self._thread = threading.Thread(
            target=self._drain, daemon=True, name="acp-win32-stdin-writer"
        )
        self._thread.start()

    def _drain(self) -> None:
        while True:
            data = self._queue.get()
            if data is None:
                break
            try:
                self._pipe.write(data)
                self._pipe.flush()
            except (OSError, ValueError):
                # Agent exited and closed its end; nothing useful to do here.
                break
        try:
            self._pipe.close()
        except (OSError, ValueError):
            pass

    def write(self, data: bytes) -> None:
        if not self._closing:
            self._queue.put(bytes(data))

    def can_write_eof(self) -> bool:
        return False

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        if not self._closing:
            self._closing = True
            self._queue.put(None)

    def abort(self) -> None:
        self.close()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default


class WindowsProcess:
    """
    An `asyncio.subprocess.Process` work-alike backed by `subprocess.Popen`.

    `stdout` is a genuine `asyncio.StreamReader`, which matters because the
    ACP SDK drives it with `readuntil(b"...")` and `readexactly(n)` and relies
    on their buffering semantics. Reproducing those correctly by hand is not
    worth it when the real class works once data is fed in from a thread.
    """

    def __init__(
        self,
        popen: subprocess.Popen,
        loop: asyncio.AbstractEventLoop,
        *,
        limit: int = DEFAULT_STREAM_LIMIT,
    ) -> None:
        self._popen = popen
        self._loop = loop
        self._wait_future: Optional[asyncio.Future] = None

        self._stdout: Optional[asyncio.StreamReader] = None
        if popen.stdout is not None:
            self._stdout = asyncio.StreamReader(limit=limit, loop=loop)
            threading.Thread(
                target=self._pump,
                args=(popen.stdout, self._stdout),
                daemon=True,
                name="acp-win32-stdout-reader",
            ).start()

        self._stdin: Optional[asyncio.StreamWriter] = None
        if popen.stdin is not None:
            transport = _WriteTransport(popen.stdin)
            protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader(loop=loop))
            self._stdin = asyncio.StreamWriter(transport, protocol, None, loop)

        # `stderr` is never piped by this package; it is inherited or merged
        # into stdout. Exposed as None to match `Process`.
        self._stderr: Optional[asyncio.StreamReader] = None

    def _pump(self, pipe: Any, reader: asyncio.StreamReader) -> None:
        """Feed a blocking pipe into an asyncio StreamReader from a thread."""
        try:
            while True:
                data = pipe.read(65536)
                if not data:
                    break
                self._loop.call_soon_threadsafe(reader.feed_data, data)
        except (OSError, ValueError):
            pass
        finally:
            self._loop.call_soon_threadsafe(reader.feed_eof)

    @property
    def stdin(self) -> Optional[asyncio.StreamWriter]:
        return self._stdin

    @property
    def stdout(self) -> Optional[asyncio.StreamReader]:
        return self._stdout

    @property
    def stderr(self) -> Optional[asyncio.StreamReader]:
        return self._stderr

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def returncode(self) -> Optional[int]:
        # `Popen.returncode` is only populated by poll()/wait(); poll here so
        # callers that gate on `returncode is None` see the truth.
        return self._popen.poll()

    async def wait(self) -> int:
        """Wait for the process to exit, without blocking the event loop."""
        if self._wait_future is None:
            self._wait_future = self._loop.run_in_executor(None, self._popen.wait)
        return await asyncio.shield(self._wait_future)

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()


def _taskkill_tree(pid: int) -> None:
    """Kill `pid` and every process it spawned."""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
        check=False,
    )


async def kill_process_tree(process: AnyProcess) -> None:
    """
    Terminate `process` and all of its descendants.

    On Windows an npm shim is really `cmd.exe` wrapping `node`, so killing only
    the direct child orphans the agent. `taskkill /T` walks the tree. On POSIX
    this is the existing `killpg` behaviour.
    """
    if process.returncode is not None:
        return

    if IS_WINDOWS:
        # `taskkill` is a subprocess itself, so keep it off the event loop.
        await asyncio.get_running_loop().run_in_executor(
            None, _taskkill_tree, process.pid
        )
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


async def terminate_process(
    process: AnyProcess, *, timeout: float = 5.0
) -> None:
    """
    Stop `process` gracefully, escalating to a forced tree kill on timeout.

    POSIX gets the original SIGINT/SIGTERM-then-SIGKILL sequence over the
    process group. Windows has no equivalent signals for a non-console child,
    so it goes straight to a tree kill.
    """
    if process.returncode is not None:
        return

    if IS_WINDOWS:
        # Order matters. An npm shim is `cmd.exe` wrapping `node`, and Windows
        # does not re-parent or signal descendants: terminating the shim first
        # orphans the agent, and once the shim has exited its pid no longer
        # names a tree that `taskkill /T` can walk. So kill the whole tree
        # while the shim is still alive, then reap it.
        await kill_process_tree(process)
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return

    pgid = os.getpgid(process.pid)
    os.killpg(pgid, signal.SIGINT)
    os.killpg(pgid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        os.killpg(pgid, signal.SIGKILL)
