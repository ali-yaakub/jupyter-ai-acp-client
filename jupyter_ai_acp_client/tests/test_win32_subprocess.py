"""
Tests for the Windows subprocess compatibility layer.

The platform-independent behaviour is asserted everywhere; the Windows-specific
paths are skipped elsewhere. The most important test here is
`test_spawn_on_selector_loop`, which reproduces the exact condition that made
every ACP persona unusable on Windows: `jupyter_server` runs the server on a
`SelectorEventLoop`, which cannot create subprocesses at all.
"""

import asyncio
import subprocess
import sys
import threading

import pytest

from jupyter_ai_acp_client._win32_subprocess import (
    IS_WINDOWS,
    WindowsProcess,
    create_subprocess,
    resolve_executable,
    terminate_process,
)

windows_only = pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific behaviour")

# Reads lines from stdin and echoes them back with a prefix, unbuffered.
ECHO_SCRIPT = (
    "import sys\n"
    "for line in sys.stdin.buffer:\n"
    "    sys.stdout.buffer.write(b'echo:' + line)\n"
    "    sys.stdout.buffer.flush()\n"
)

# Writes one newline-terminated line, then stays alive so a reader that waits
# for a *full* buffer rather than a line will visibly block.
# Spawns a child of its own, reports the child's pid, then idles — standing in
# for the `cmd.exe` -> `node` shape of an npm-installed ACP adapter.
SPAWNER_SCRIPT = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "sys.stdout.buffer.write(str(child.pid).encode() + b'\\n')\n"
    "sys.stdout.buffer.flush()\n"
    "time.sleep(300)\n"
)

SLOW_SCRIPT = (
    "import sys, time\n"
    "sys.stdout.buffer.write(b'first\\n')\n"
    "sys.stdout.buffer.flush()\n"
    "time.sleep(30)\n"
)


# ===================================================================
# resolve_executable
# ===================================================================

class TestResolveExecutable:
    def test_preserves_extra_args(self):
        args = resolve_executable(["claude-agent-acp", "--flag", "value"])
        assert args[1:] == ["--flag", "value"]

    def test_leaves_explicit_paths_alone(self):
        """An explicit path is the caller's choice; don't second-guess it."""
        explicit = str(sys.executable)
        assert resolve_executable([explicit])[0] == explicit

    def test_unknown_command_passes_through(self):
        """
        An unresolvable command keeps its original spelling so the caller still
        gets a FileNotFoundError naming what the user actually configured.
        """
        assert resolve_executable(["definitely-not-a-real-command-xyz"]) == [
            "definitely-not-a-real-command-xyz"
        ]

    def test_empty_argv(self):
        assert resolve_executable([]) == []

    @windows_only
    def test_resolves_bare_name_to_full_path(self):
        """
        The `.cmd` bug: CreateProcessW only appends `.exe` when resolving a bare
        name, so npm shims are invisible to it. `shutil.which` honours PATHEXT.
        """
        resolved = resolve_executable(["cmd"])[0]
        assert resolved.lower().endswith(".exe")
        assert len(resolved) > len("cmd")

    @pytest.mark.skipif(IS_WINDOWS, reason="POSIX-specific")
    def test_noop_on_posix(self):
        """Resolution is a Windows workaround; POSIX argv is untouched."""
        assert resolve_executable(["sh", "-c", "true"]) == ["sh", "-c", "true"]


# ===================================================================
# create_subprocess
# ===================================================================

class TestCreateSubprocess:
    async def test_spawns_and_streams(self):
        proc = await create_subprocess(
            sys.executable,
            "-u",
            "-c",
            ECHO_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
        )
        try:
            proc.stdin.write(b"hello\n")
            await proc.stdin.drain()
            assert await asyncio.wait_for(proc.stdout.readuntil(b"\n"), 30) == b"echo:hello\n"
        finally:
            proc.kill()
            await proc.wait()

    async def test_uses_native_process_when_loop_supports_it(self):
        """
        The bridge is a fallback, not the default: a loop that can spawn
        subprocesses should still yield a real `asyncio.subprocess.Process`.
        """
        proc = await create_subprocess(
            sys.executable, "-c", "pass", stdout=asyncio.subprocess.PIPE
        )
        try:
            if IS_WINDOWS and isinstance(
                asyncio.get_running_loop(), asyncio.SelectorEventLoop
            ):
                pytest.skip("test loop cannot spawn subprocesses natively")
            assert isinstance(proc, asyncio.subprocess.Process)
        finally:
            await proc.wait()


@windows_only
class TestSelectorEventLoopFallback:
    """
    Regression tests for the original bug.

    `jupyter_server` downgrades Windows to a `SelectorEventLoop`
    (`ServerApp._init_asyncio_patch`), on which `create_subprocess_exec` raises
    `NotImplementedError`. Every ACP persona died here.
    """

    def test_asyncio_really_cannot_spawn_here(self):
        """Guard the premise: if this ever starts passing, the bridge is moot."""
        loop = asyncio.SelectorEventLoop()
        try:
            with pytest.raises(NotImplementedError):
                loop.run_until_complete(
                    asyncio.create_subprocess_exec(
                        sys.executable, "-c", "pass", stdout=asyncio.subprocess.PIPE
                    )
                )
        finally:
            loop.close()

    def test_spawn_on_selector_loop(self):
        """The whole point: an ACP agent must start on the server's own loop."""
        loop = asyncio.SelectorEventLoop()

        async def go():
            proc = await create_subprocess(
                sys.executable,
                "-u",
                "-c",
                ECHO_SCRIPT,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,
            )
            assert isinstance(proc, WindowsProcess)
            try:
                # ACP frames are newline-delimited JSON, so `readuntil` is the
                # call that actually has to work.
                proc.stdin.write(b'{"jsonrpc":"2.0"}\n')
                await proc.stdin.drain()
                line = await asyncio.wait_for(proc.stdout.readuntil(b"\n"), 30)
                assert line == b'echo:{"jsonrpc":"2.0"}\n'
                assert proc.pid > 0
            finally:
                proc.kill()
                await proc.wait()

        try:
            loop.run_until_complete(go())
        finally:
            loop.close()

    def test_devnull_stdin_is_not_silently_made_a_pipe(self):
        """
        `create_terminal` asks for DEVNULL stdin. If that were translated into
        a pipe, any command reading stdin would block forever instead of
        seeing EOF.
        """
        loop = asyncio.SelectorEventLoop()

        async def go():
            proc = await create_subprocess(
                sys.executable,
                "-u",
                "-c",
                "import sys; sys.stdout.buffer.write("
                "b'read:' + sys.stdin.read().encode() + b'|\\n')",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,
            )
            try:
                assert proc.stdin is None
                line = await asyncio.wait_for(proc.stdout.readuntil(b"\n"), 30)
                # Empty payload => the child read EOF rather than blocking.
                assert line == b"read:|\n"
            finally:
                proc.kill()
                await proc.wait()

        try:
            loop.run_until_complete(go())
        finally:
            loop.close()

    def test_stderr_can_be_merged_into_stdout(self):
        """`create_terminal` merges stderr into stdout via the STDOUT sentinel."""
        loop = asyncio.SelectorEventLoop()

        async def go():
            proc = await create_subprocess(
                sys.executable,
                "-u",
                "-c",
                "import sys; sys.stderr.buffer.write(b'from-stderr\\n')",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                line = await asyncio.wait_for(proc.stdout.readuntil(b"\n"), 30)
                assert line == b"from-stderr\n"
            finally:
                proc.kill()
                await proc.wait()

        try:
            loop.run_until_complete(go())
        finally:
            loop.close()

    def test_returncode_reflects_exit(self):
        loop = asyncio.SelectorEventLoop()

        async def go():
            proc = await create_subprocess(
                sys.executable,
                "-c",
                "import sys; sys.exit(3)",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,
            )
            assert await asyncio.wait_for(proc.wait(), 30) == 3
            assert proc.returncode == 3

        try:
            loop.run_until_complete(go())
        finally:
            loop.close()


@windows_only
class TestTreeTermination:
    """
    Regression test for orphaned agents.

    An npm shim is `cmd.exe` wrapping `node`, so the real agent is a
    *grandchild* of what we spawned. Windows neither re-parents nor signals
    descendants, so terminating the direct child leaves the agent running.
    This was observable as a `node.exe` surviving a JupyterLab shutdown.
    """

    @staticmethod
    def _alive(pid: int) -> bool:
        out = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % pid],
            capture_output=True,
            text=True,
        ).stdout
        return str(pid) in out

    def test_terminate_process_kills_the_grandchild(self):
        loop = asyncio.SelectorEventLoop()

        async def go():
            proc = await create_subprocess(
                sys.executable,
                "-u",
                "-c",
                SPAWNER_SCRIPT,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,
            )
            line = await asyncio.wait_for(proc.stdout.readuntil(b"\n"), 30)
            grandchild = int(line.strip())
            assert self._alive(grandchild), "grandchild never started"

            await terminate_process(proc, timeout=15)

            # Windows reaps asynchronously; give it a moment to settle.
            for _ in range(50):
                if not self._alive(grandchild):
                    break
                await asyncio.sleep(0.1)
            assert not self._alive(grandchild), (
                "grandchild %d survived termination; the process tree was not "
                "killed while the direct child was still alive" % grandchild
            )

        try:
            loop.run_until_complete(go())
        finally:
            loop.close()


class TestPipeBufferingAssumption:
    """
    The bridge's reader thread calls `pipe.read(n)`, which is only safe because
    `Popen` is given `bufsize=0`. With default buffering the same call blocks
    until `n` bytes or EOF, which deadlocks against a request/response protocol
    where the agent is waiting for our reply.

    This test exists so that anyone who "tidies up" `bufsize` sees why not.
    """

    @staticmethod
    def _first_read_returns(bufsize: int) -> bool:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", SLOW_SCRIPT],
            stdout=subprocess.PIPE,
            bufsize=bufsize,
        )
        done = threading.Event()
        try:
            threading.Thread(
                target=lambda: (proc.stdout.read(65536), done.set()),
                daemon=True,
            ).start()
            return done.wait(timeout=10)
        finally:
            proc.kill()
            proc.wait()

    def test_unbuffered_pipe_yields_a_partial_read(self):
        assert self._first_read_returns(0), (
            "raw pipe read should return as soon as bytes arrive"
        )

    def test_buffered_pipe_would_block(self):
        assert not self._first_read_returns(-1), (
            "buffered read returned early; if this now streams, revisit the "
            "bufsize=0 requirement in _create_subprocess_via_popen"
        )
