# Tool execution

- Run project commands from the repository root after activating `.venv`.
- In PowerShell, initialize UTF-8 before running Python-based tools:

  ```powershell
  . .\.venv\Scripts\Activate.ps1
  $env:PYTHONUTF8 = '1'
  $env:PYTHONIOENCODING = 'utf-8'
  ```

- Run `pytest` with `sandbox_permissions="require_escalated"` on the first invocation. The Windows sandbox cannot access the existing pytest temporary directory or `.pytest_cache`. Do not repeat a sandboxed attempt to rediscover this known access failure.
- Keep pytest defaults from `pyproject.toml`; use only the test paths and options needed for the task.
- Set Python encoding in the command environment. Do not add pytest hooks or application wrappers to reconfigure standard streams.
- A sandbox `EPERM` while a project command reads a tool installed outside the repository is not a broken project environment. Rerun the same command with sandbox escalation so the tool can read its installation.
- In particular, `pyright` may need sandbox escalation to read `C:\Users\hunte\AppData\Roaming\uv\tools\pyright`. Do not replace it with a path-based invocation or report that Pyright could not run before retrying the activated-environment command with escalation.
