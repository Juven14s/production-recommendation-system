"""Atomic local deployment registry complementing the MLflow registry."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class LocalModelRegistry:
    """Track deployable bundles and champion history in one atomic JSON file."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.bundles_dir = self.root / "bundles"
        self.state_path = self.root / "registry.json"
        self.bundles_dir.mkdir(parents=True, exist_ok=True)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "champion": None,
            "previous_champion": None,
            "versions": {},
            "history": [],
        }

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty_state()
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.state_path)

    def register(self, version: str, metadata: dict[str, Any]) -> None:
        state = self.load()
        if version in state["versions"]:
            raise ValueError(f"model version already exists: {version}")
        state["versions"][version] = metadata
        state["history"].append(
            {"action": "register", "version": version, "at": utc_now()}
        )
        self._write(state)

    def promote(self, version: str) -> None:
        state = self.load()
        if version not in state["versions"]:
            raise KeyError(f"unknown model version: {version}")
        previous = state["champion"]
        state["previous_champion"] = previous
        state["champion"] = version
        state["versions"][version]["status"] = "champion"
        if previous and previous in state["versions"]:
            state["versions"][previous]["status"] = "archived"
        state["history"].append(
            {
                "action": "promote",
                "from": previous,
                "to": version,
                "at": utc_now(),
            }
        )
        self._write(state)

    def rollback(self, target_version: str | None = None) -> str:
        state = self.load()
        current = state["champion"]
        target = target_version or state["previous_champion"]
        if target is None:
            raise ValueError("no previous champion is available for rollback")
        if target not in state["versions"]:
            raise KeyError(f"unknown rollback target: {target}")
        state["champion"] = target
        state["previous_champion"] = current
        state["versions"][target]["status"] = "champion"
        if current and current in state["versions"]:
            state["versions"][current]["status"] = "rolled_back"
        state["history"].append(
            {
                "action": "rollback",
                "from": current,
                "to": target,
                "at": utc_now(),
            }
        )
        self._write(state)
        return target

    def champion_bundle(self) -> Path:
        state = self.load()
        champion = state["champion"]
        if champion is None:
            raise ValueError("the registry has no champion model")
        relative_path = state["versions"][champion]["bundle_path"].replace("\\", "/")
        bundle = (self.root / relative_path).resolve()
        if not bundle.is_relative_to(self.root):
            raise ValueError("registered bundle path escapes the registry root")
        if not bundle.is_dir():
            raise FileNotFoundError(f"champion bundle is missing: {bundle}")
        return bundle
