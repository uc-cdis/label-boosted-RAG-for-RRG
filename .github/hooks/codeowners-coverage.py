#!/usr/bin/env python3
"""
Pre-commit hook: verify CODEOWNERS and the git-tracked file tree are in sync.

  1. Every tracked file must be matched by at least one CODEOWNERS pattern.
  2. Every CODEOWNERS pattern must match at least one tracked file.

Managed via pre-commit. To run manually:
    python3 .github/hooks/codeowners-coverage.py
"""
import fnmatch
import os
import subprocess
import sys

CODEOWNERS_PATH = ".github/CODEOWNERS"


def parse_patterns(path: str) -> list[str]:
    """Return CODEOWNERS patterns in file order (last match wins semantics)."""
    patterns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line.split()[0])
    return patterns


def pattern_matches(pattern: str, file: str) -> bool:
    """Return True if a single CODEOWNERS pattern matches the given file path."""
    if pattern in ("*", "/**"):
        return True

    anchored = pattern.startswith("/")
    norm = pattern.lstrip("/")

    if norm.endswith("/"):
        return file.startswith(norm) or fnmatch.fnmatch(file, norm + "*")

    if not anchored and "/" not in norm:
        return fnmatch.fnmatch(os.path.basename(file), norm)

    return fnmatch.fnmatch(file, norm) or file.startswith(norm.rstrip("/") + "/")


def has_owner(file: str, patterns: list[str]) -> bool:
    """Return True if the file is matched by any CODEOWNERS pattern."""
    return any(pattern_matches(p, file) for p in patterns)


def pattern_is_stale(pattern: str, files: list[str]) -> bool:
    """Return True if no tracked file matches this pattern."""
    return not any(pattern_matches(pattern, f) for f in files)


def tracked_files() -> list[str]:
    """All files currently tracked by git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.splitlines() if f]


def main() -> None:
    patterns = parse_patterns(CODEOWNERS_PATH)
    files = tracked_files()

    missing_owner = [f for f in files if not has_owner(f, patterns)]
    stale_patterns = [p for p in patterns if pattern_is_stale(p, files)]

    if not missing_owner and not stale_patterns:
        print("CODEOWNERS is in sync with the repository.")
        return

    if missing_owner:
        print("error: the following files have no CODEOWNERS entry:", file=sys.stderr)
        for f in missing_owner:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\nAdd a pattern covering each file to {CODEOWNERS_PATH}", file=sys.stderr
        )

    if stale_patterns:
        if missing_owner:
            print("", file=sys.stderr)
        print(
            "error: the following CODEOWNERS patterns match no tracked files:",
            file=sys.stderr,
        )
        for p in stale_patterns:
            print(f"  {p}", file=sys.stderr)
        print(
            f"\nRemove or update these patterns in {CODEOWNERS_PATH}", file=sys.stderr
        )

    sys.exit(1)


if __name__ == "__main__":
    main()
