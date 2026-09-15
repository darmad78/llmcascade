"""In-place model ID edits in models.yaml (preserve comments / ordering)."""

from __future__ import annotations

import re


def replace_model_id(text: str, old: str, new: str) -> str:
    if old == new:
        return text
    esc = re.escape(old)
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if re.match(rf'(?:-\s+)?name:\s*["\']?{esc}["\']?\s*$', stripped):
            out.append(re.sub(esc, new, line, count=1))
            continue
        if re.match(rf'api_model:\s*["\']?{esc}["\']?\s*$', stripped):
            out.append(re.sub(esc, new, line, count=1))
            continue
        if re.match(rf'- ["\']?{esc}["\']?\s*$', stripped):
            out.append(re.sub(esc, new, line, count=1))
            continue
        if "free_tier_note:" in stripped and old in line:
            out.append(line.replace(old, new))
            continue
        out.append(line)
    return "".join(out)
