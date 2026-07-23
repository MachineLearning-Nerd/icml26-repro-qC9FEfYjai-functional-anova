"""Emit an installed-package inventory when uv created an unseeded virtualenv."""

from __future__ import annotations

import importlib.metadata
import sys


def main() -> int:
    if sys.argv[1:] != ["freeze"]:
        print("This compatibility shim supports only: python -m pip freeze", file=sys.stderr)
        return 2

    by_name: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        version = distribution.version
        if name and version:
            by_name[str(name).casefold()] = (str(name), str(version))
    installed = sorted(by_name.values(), key=lambda item: item[0].casefold())
    for name, version in installed:
        print(f"{name}=={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
