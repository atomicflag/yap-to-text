# AGENTS.md — Project Guidelines for AI Agents

## Python Import Policy

Python `import` statements are **only allowed at the top of a module** (top-level scope).

The following locations are **forbidden**:

- Inside functions or methods
- Inside classes (including class body)
- Inside `if`/`else`/`try`/`with` blocks, even when nested within top-level code
- Inside comprehensions or lambda expressions

Every import must appear at the module level, ordered and grouped according to your project's existing style.

## Nested Definitions Policy

Defining **functions** or **classes** inside other functions is **forbidden**. All function and class definitions must be at the module level.

## Logging Policy

Python `print` statements are **forbidden**. Use standard Python logging (`logging`) instead.

The `logger = logging.getLogger(__name__)` pattern is allowed, but it **MUST** be placed at the top of the module, after all import statements and before any other code (functions, classes, or executable logic).

## Running the App

The agent is **only allowed to run the app with `--help`** parameter, which displays help information and exits. **Launching the app normally is forbidden**, as it requires a large amount of VRAM.

To test application changes, rely on the `ruff` and `ty` checks described in the Validation Checks section below.

## Global Variables Policy

Global (module-level) non-constant variables are **forbidden**. The only exceptions are idiomatic initialization patterns from popular libraries, such as:

- `app = FastAPI()`
- `app = typer.Typer()`
- `logger = logging.getLogger(__name__)`

The `global` keyword is also **forbidden**.

## Validation Checks

For a change to be considered **successful**, it **must pass both** of the following checks:

1. Run `ruff format .` to ensure formatting compliance.
2. Then run:

```bash
ruff check --fix .
ty check .
```

If either command reports errors or warnings, the change is not complete and must be fixed before submission.

## Package & Dependency Management Policy

The project uses `uv` for all package and dependency management. The use of raw `pip` commands (e.g. `pip install`, `pip freeze`, `pip uninstall`) is **forbidden**. Use `uv add` / `uv remove` for managing project dependencies in `pyproject.toml`

When adding, removing, or updating project dependencies — **always ask the user for permission** before making changes.

## Comment Style Policy

Comments must be plain and simple. The following are **forbidden**:

- Lines, borders, or separators made of characters like `=`, `-`, `*`, `#`
- Emojis
- ASCII art or decorative characters

Only use straightforward, readable text in comments.
