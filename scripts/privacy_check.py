#!/usr/bin/env python3
"""Fail the build if identifying information leaks into the repository.

    python scripts/privacy_check.py

Two layers:

  1. **Patterns** — email, mainland mobile, resident id, absolute home path. Always on,
     so the script is useful to anyone who forks it.
  2. **Literal terms** — read from ``.privacy-terms.txt`` (one term per line, gitignored).
     The terms themselves must never live in a public file: a checker that lists what it
     is hiding is a map of what to look for. The file ships as ``.privacy-terms.example.txt``
     with placeholder entries.

Exit code 1 on any hit, so it can be wired into a pre-commit hook or CI.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TERMS_FILE = ROOT / ".privacy-terms.txt"

# The public contact address is intentional.
ALLOW_EMAILS = {"m4rh0zen@foxmail.com"}
ALLOW_PATHS = {"/Users/<user>", "/Users/example"}

PATTERNS = [
    # bare emails, but not version strings or decorators like @media
    (re.compile(r"(?<![\w.@])[\w.+-]+@[\w-]+\.[\w.]{2,}"), "email address"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "mainland mobile number"),
    # 18-digit resident id: needs a plausible birth date and not to be a decimal
    (re.compile(r"(?<![.\d])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![.\d])"),
     "resident id number"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+"), "absolute macOS home path"),
    (re.compile(r"/home/[A-Za-z0-9._-]+/"), "absolute Linux home path"),
]

SKIP_SUFFIXES = {".pt", ".parquet", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff2", ".npz"}
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}


def load_terms() -> list[str]:
    if not TERMS_FILE.exists():
        return []
    return [
        line.strip()
        for line in TERMS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        paths = [ROOT / p for p in out]
    except (subprocess.CalledProcessError, FileNotFoundError):
        paths = [p for p in ROOT.rglob("*") if p.is_file()]
    return [p for p in paths if p.exists()]


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in {"scripts/privacy_check.py", ".privacy-terms.txt", ".gitignore"}:
        return True
    if any(rel.startswith(d + "/") for d in SKIP_DIRS):
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def scan() -> list[str]:
    terms = load_terms()
    problems: list[str] = []
    for path in tracked_files():
        if should_skip(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            for term in terms:
                if term in line:
                    problems.append(f"{rel}:{lineno}: literal term {term!r}")
            for pattern, label in PATTERNS:
                for match in pattern.finditer(line):
                    hit = match.group(0)
                    if label == "email address" and hit in ALLOW_EMAILS:
                        continue
                    if label == "absolute macOS home path" and hit in ALLOW_PATHS:
                        continue
                    problems.append(f"{rel}:{lineno}: {label} -> {hit}")
    return problems


def main() -> int:
    terms = load_terms()
    if not terms:
        print("privacy_check: no .privacy-terms.txt, running pattern checks only")
    problems = scan()
    if not problems:
        print("privacy_check: clean")
        return 0
    print(f"privacy_check: {len(problems)} issue(s) found")
    for p in problems:
        print("  " + p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
