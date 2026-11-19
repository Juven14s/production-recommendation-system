"""Create the minimal local registry shipped in a release image."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def write_release_registry(bundle: Path, output: Path) -> dict[str, object]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    version = str(manifest["version"])
    state: dict[str, object] = {
        "schema_version": 1,
        "champion": version,
        "previous_champion": None,
        "versions": {
            version: {
                "bundle_path": "bundles/champion",
                "created_at": manifest["created_at"],
                "run_id": manifest["run_id"],
                "status": "champion",
                "metrics": manifest["metrics"],
            }
        },
        "history": [
            {
                "action": "release_image",
                "to": version,
                "at": datetime.now(UTC).isoformat(),
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a release image registry.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_release_registry(args.bundle.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
