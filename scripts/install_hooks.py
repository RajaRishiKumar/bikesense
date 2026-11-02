"""Install the secret-check pre-commit hook into .git/hooks (run once after cloning)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
hook = ROOT / ".git" / "hooks" / "pre-commit"
hook.parent.mkdir(parents=True, exist_ok=True)
hook.write_text(
    "#!/bin/sh\n"
    "# BikeSense: block commits containing API keys or .env files\n"
    'PY="$(git rev-parse --show-toplevel)/.venv/Scripts/python.exe"\n'
    '[ -x "$PY" ] || PY="$(git rev-parse --show-toplevel)/.venv/bin/python"\n'
    '[ -x "$PY" ] || PY=python\n'
    '"$PY" scripts/check_secrets.py\n',
    encoding="utf-8",
    newline="\n",
)
hook.chmod(0o755)
print(f"installed {hook}")
