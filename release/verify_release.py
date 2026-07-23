#!/usr/bin/env python3
"""Fail-closed checks for the additive, text-only Space overlay."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPDATES = ROOT / "space_updates"
ALLOWLIST = ROOT / "upload_allowlist.txt"
MANIFEST = ROOT / "protected_space_manifest.json"

EXPECTED_OLD_CHILDREN = [
    ("overview", "pages/overview/page.md"),
    (
        "claim-1-closed-form-decomposition",
        "pages/claim-1-closed-form-decomposition/page.md",
    ),
    (
        "claim-2-efficient-no-sampling",
        "pages/claim-2-efficient-no-sampling/page.md",
    ),
    (
        "claim-3-non-rectangular-dependent-support",
        "pages/claim-3-non-rectangular-dependent-support/page.md",
    ),
    ("methods-environment", "pages/methods-environment/page.md"),
    ("conclusion", "pages/conclusion/page.md"),
]
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token)\\s*[:=]\\s*['\"][^'\"]+"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protected = json.loads(MANIFEST.read_text())
    assert protected["space_id"] == "DineshAI/qC9FEfYjai"
    assert (
        protected["revision"]
        == "6ce8ca3f9b6bd4071976f04c7b105db70e5c5e92"
    )
    old_paths = {row["path"] for row in protected["files"]}

    allowed = [
        line.strip()
        for line in ALLOWLIST.read_text().splitlines()
        if line.strip()
    ]
    update_paths = sorted(
        str(path.relative_to(UPDATES))
        for path in UPDATES.rglob("*")
        if path.is_file()
    )
    assert sorted(allowed) == update_paths
    assert len(allowed) == len(set(allowed))
    assert set(allowed) & old_paths == {"logbook.json"}

    for relative in allowed:
        path = UPDATES / relative
        text = path.read_text(encoding="utf-8")
        assert "\x00" not in text
        assert not any(pattern.search(text) for pattern in SECRET_PATTERNS)

    logbook = json.loads((UPDATES / "logbook.json").read_text())
    assert logbook["schema_version"] == 1
    assert logbook["space_id"] == protected["space_id"]
    children = logbook["root"]["children"]
    observed_old = [(child["slug"], child["file"]) for child in children[:6]]
    assert observed_old == EXPECTED_OLD_CHILDREN
    assert all(child["file"] in old_paths for child in children[:6])
    assert all(
        (UPDATES / child["file"]).is_file()
        for child in children[6:]
    )

    generated = [
        {
            "bytes": (UPDATES / relative).stat().st_size,
            "path": relative,
            "sha256": sha256(UPDATES / relative),
        }
        for relative in allowed
    ]
    expected = json.loads((ROOT / "upload_manifest.json").read_text())
    assert expected["files"] == generated
    assert expected["space_id"] == protected["space_id"]
    assert expected["base_revision"] == protected["revision"]

    print(
        json.dumps(
            {
                "candidate_logbook_valid": True,
                "protected_old_file_count": len(old_paths),
                "old_paths_preserved_by_overlay": True,
                "old_navigation_entries_preserved": True,
                "text_only_upload_paths": len(allowed),
                "secret_scan_passed": True,
                "upload_manifest_matches": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
