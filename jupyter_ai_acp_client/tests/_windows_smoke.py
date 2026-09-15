"""
End-to-end Windows smoke test, run from CI rather than under pytest.

The unit tests exercise the subprocess bridge against `sys.executable`. This
script exercises it against a real npm-installed ACP adapter, on the same kind
of event loop `jupyter_server` gives the extension, and drives it with the real
ACP SDK. It therefore covers both Windows defects at once:

  1. `SelectorEventLoop` cannot spawn subprocesses      -> NotImplementedError
  2. `CreateProcessW` only resolves `.exe`, so a bare
     command name never finds an npm `.cmd` shim        -> FileNotFoundError

Passing an unresolved bare name here is deliberate: it is exactly what the
persona classes do, and it is what defect 2 broke.

Exits non-zero on failure.
"""

import asyncio
import os
import sys

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.schema import ClientCapabilities, FileSystemCapabilities

from jupyter_ai_acp_client._win32_subprocess import create_subprocess

AGENT = "claude-agent-acp"


class _NullClient:
    """Minimal ACP client: `initialize` needs no callbacks to come back."""

    async def request_permission(self, *args, **kwargs):
        raise NotImplementedError

    async def session_update(self, *args, **kwargs):
        return None

    async def write_text_file(self, *args, **kwargs):
        raise NotImplementedError

    async def read_text_file(self, *args, **kwargs):
        raise NotImplementedError

    async def create_terminal(self, *args, **kwargs):
        raise NotImplementedError

    async def terminal_output(self, *args, **kwargs):
        raise NotImplementedError

    async def wait_for_terminal_exit(self, *args, **kwargs):
        raise NotImplementedError

    async def kill_terminal(self, *args, **kwargs):
        raise NotImplementedError

    async def release_terminal(self, *args, **kwargs):
        raise NotImplementedError

    async def ext_method(self, *args, **kwargs):
        raise NotImplementedError

    async def ext_notification(self, *args, **kwargs):
        return None


async def main() -> None:
    loop = asyncio.get_running_loop()
    print("event loop:", type(loop).__name__)
    assert not hasattr(loop, "_proactor"), (
        "expected a SelectorEventLoop; this test is meaningless on a Proactor loop"
    )

    proc = await create_subprocess(
        AGENT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
        limit=50 * 1024 * 1024,
    )
    print("spawned %r as pid %s via %s" % (AGENT, proc.pid, type(proc).__name__))

    try:
        conn = connect_to_agent(_NullClient(), proc.stdin, proc.stdout)
        response = await asyncio.wait_for(
            conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapabilities(
                        read_text_file=False, write_text_file=False
                    )
                ),
            ),
            timeout=120,
        )
        print("ACP initialize OK; protocol version", response.protocol_version)
    finally:
        proc.kill()
        await proc.wait()


if __name__ == "__main__":
    if sys.platform != "win32":
        print("skipped: Windows only")
        raise SystemExit(0)

    # Reproduce what jupyter_server does to the event loop on Windows.
    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    print("Windows ACP smoke test passed.")
