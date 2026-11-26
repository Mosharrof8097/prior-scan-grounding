#!/usr/bin/env python3
"""
tests/test_forbidden_phrases.py

Automated test asserting that none of the unscientific / sensationalized forbidden phrases
appear in any markdown (.md) documentation file across the repository.
"""

import os
import unittest
from typing import List, Tuple

FORBIDDEN_PHRASES = [
    "smoking gun",
    "ultimate",
    "universal vision blindness",
    "confirmed blind",
    "100% proven",
    "negative evidence",
    "pure language prior",
    "actively penalizes confidence",
    "textbook prior"
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE_DIRS = {".git", "thesis_env", "venv", ".system_generated", "brain", "__pycache__"}


class TestForbiddenPhrases(unittest.TestCase):
    def test_no_forbidden_phrases_in_markdown(self):
        violations: List[Tuple[str, int, str, str]] = []

        for root, dirs, files in os.walk(REPO_ROOT):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if file.endswith(".md"):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, REPO_ROOT)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            for line_num, line in enumerate(f, 1):
                                line_lower = line.lower()
                                for phrase in FORBIDDEN_PHRASES:
                                    if phrase in line_lower:
                                        violations.append((rel_path, line_num, phrase, line.strip()))
                    except Exception as e:
                        self.fail(f"Could not read {filepath}: {e}")

        if violations:
            msg_lines = [f"Found {len(violations)} forbidden phrase violations in documentation:"]
            for rel_path, line_num, phrase, line in violations:
                msg_lines.append(f"  {rel_path}:{line_num} contains '{phrase}' -> \"{line}\"")
            self.fail("\n".join(msg_lines))


if __name__ == "__main__":
    unittest.main()
