#!/bin/bash
# SessionStart hook: prepare the environment so tests and linters can run in
# Claude Code on the web sessions.
#
# Runs synchronously, so the session does not start until dependencies are in
# place. That costs a little startup latency but removes the race where the
# agent tries to run tests before the venv exists.
set -euo pipefail

# Local machines already have a working checkout; only the remote containers
# start from scratch.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"

# Poetry lands in ~/.local/bin, which is not always on PATH in a fresh container.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v poetry >/dev/null 2>&1; then
  echo "session-start: poetry not found, installing from PyPI" >&2
  python3 -m pip install --user --quiet poetry
  export PATH="$HOME/.local/bin:$PATH"
fi

# poetry.toml pins virtualenvs.in-project, so this creates ./.venv (gitignored).
# Plain `install` rather than a locked sync: the container image is cached after
# this hook completes, and `install` is a fast no-op on a warm cache.
echo "session-start: installing dependencies with poetry" >&2
poetry install

# Expose the venv to the rest of the session so `python -m unittest`, `black`
# and `pylint` work without a `poetry run` prefix.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=\"$PROJECT_DIR/.venv\""
    echo "export PATH=\"$PROJECT_DIR/.venv/bin:\$PATH\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "session-start: ready" >&2
