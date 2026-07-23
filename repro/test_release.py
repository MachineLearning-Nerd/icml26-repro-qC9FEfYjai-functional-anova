import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_additive_space_release_gate():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "release" / "verify_release.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["candidate_logbook_valid"]
    assert result["old_paths_preserved_by_overlay"]
    assert result["old_navigation_entries_preserved"]
    assert result["secret_scan_passed"]
    assert result["upload_manifest_matches"]
