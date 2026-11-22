# Tool execution

- Run project commands from the repository root after activating `.venv` or another existing virtual environment.
- Run `pytest` with `sandbox_permissions="require_escalated"` on the first invocation. The Windows sandbox cannot access the existing pytest temporary directory or `.pytest_cache`. Do not repeat a sandboxed attempt to rediscover this known access failure.
- Keep pytest defaults from `pyproject.toml`; use only the test paths and options needed for the task.
- Do not add pytest hooks or application wrappers to reconfigure standard streams.
- A sandbox `EPERM` while a project command reads a tool installed outside the repository is not a broken project environment. Rerun the same command with sandbox escalation so the tool can read its installation.
- Check types with `pyrefly check`, not `pyright`. `pyrefly` may need sandbox escalation to read `C:\Users\hunte\AppData\Roaming\uv\tools\pyrefly`. Do not replace it with a path-based invocation or report that Pyrefly could not run before retrying the activated-environment command with escalation.

# AST generation in `mapFolding.kitAST`

- Import `IngredientsFunction`, `IngredientsModule`, `LedgerOfImports`, and their helpers from `mapFolding.kitAST`. Return an `IngredientsFunction` when a generated function and its dependencies form a portable unit. Add its imports through `ingredientsFunction.imports`. Use `astModuleToIngredientsFunction` when the task calls for extracting a named function and collecting the source imports.
- Use `IngredientsModule` to assemble and write generated modules. Retain existing correct uses of the container and its prologue, functions, epilogue, and launcher. For launcher changes, preserve the other sections, public interfaces, and metadata. Extract main-guard bodies only from the supplied top-level guards and preserve nested definitions and conditionals.
- For a whole-module transformation, preserve the full transformed body, definition order, and import scope. Append the complete module to the appropriate section. Use the ledger for imports that can be consolidated without changing behavior. Leave other imports in the source body, and place new imports after any `__future__` imports. Extract individual functions only when required by the transformation.
- Use `LedgerOfImports.extractImports` to move eligible top-level imports out of an assembled section. It mutates the module body and consolidates imports from top-level, import-only `TYPE_CHECKING` blocks into the typing ledger. Preserve other conditional imports and imports inside definitions in place. Check ordering and namespace collisions after assembly, and replace or alias only conflicting bindings. Unused-import cleanup does not resolve collisions.
- Use `setDatatypeViaImport` for configured datatype import substitutions while preserving the abstract datatype names used by the generated code. Capture both returned containers. Use `toDisk` in generators that resolve their output paths from project defaults and caller overrides.
