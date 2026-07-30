# Tool execution

- Run project commands from the repository root after activating `.venv`.
- A sandbox `EPERM` while a project command reads a tool installed outside the repository is not a broken project environment. Rerun the same command with sandbox escalation so the tool can read its installation.
- In particular, `pyright` may need sandbox escalation to read `C:\Users\hunte\AppData\Roaming\uv\tools\pyright`. Do not replace it with a path-based invocation or report that Pyright could not run before retrying the activated-environment command with escalation.
