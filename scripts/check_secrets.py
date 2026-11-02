"""Pre-commit guard: block commits that would publish an API key.

Installed as .git/hooks/pre-commit by scripts/install_hooks.py. Can also be run by hand:
  python scripts/check_secrets.py          # checks staged files
  python scripts/check_secrets.py --all    # checks every tracked + untracked (non-ignored) file
"""

from __future__ import annotations

import re
import subprocess
import sys

BLOCKED_NAMES = re.compile(r"(^|/)\.env($|\.(?!example$)[^/]+$)")
PATTERNS = [
    (re.compile(r"MODEL_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-\.]{8,}"), "MODEL_API_KEY with a value"),
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"), "literal Bearer token"),
    (re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{24,}['\"]"), "hard-coded key/secret"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, encoding="utf-8").stdout


def main() -> int:
    if "--all" in sys.argv:
        files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        read = lambda f: open(f, encoding="utf-8", errors="ignore").read()  # noqa: E731
    else:
        files = git("diff", "--cached", "--name-only", "--diff-filter=ACM").splitlines()
        read = lambda f: git("show", f":{f}")  # noqa: E731  (staged content, not working copy)

    problems: list[str] = []
    for f in files:
        if BLOCKED_NAMES.search(f):
            problems.append(f"{f}: .env files must never be committed")
            continue
        if f.endswith((".png", ".jpg", ".jpeg", ".pdf", ".mp4", ".mov", ".wav", ".mp3", ".m4a")):
            continue
        try:
            text = read(f)
        except (subprocess.CalledProcessError, OSError):
            continue
        for rx, what in PATTERNS:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{f}:{line}: {what}")

    if problems:
        print("SECRET CHECK FAILED. Commit blocked:\n  " + "\n  ".join(problems))
        print("Remove the secret (keep keys only in .env), then try again.")
        return 1
    print(f"secret check: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
