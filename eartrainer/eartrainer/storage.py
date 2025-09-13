from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .models import Settings, Stats


def _data_path() -> Path:
	home = Path.home()
	dir_ = home / ".eartrainer"
	dir_.mkdir(parents=True, exist_ok=True)
	return dir_ / "data.json"


def _load_raw() -> Dict[str, Any]:
	p = _data_path()
	if not p.exists():
		return {}
	try:
		data = json.loads(p.read_text())
		return data if isinstance(data, dict) else {}
	except Exception:
		return {}


def _save_raw(data: Dict[str, Any]) -> None:
	p = _data_path()
	p.write_text(json.dumps(data, indent=2))


def load_settings() -> Settings:
	raw = _load_raw()
	obj = raw.get("settings", {})
	if isinstance(obj, dict):
		return Settings.model_validate(obj)
	return Settings()


def save_settings(s: Settings) -> None:
	raw = _load_raw()
	raw["settings"] = s.model_dump()
	_add_defaults_if_missing(raw)
	_save_raw(raw)


def load_stats() -> Stats:
	raw = _load_raw()
	obj = raw.get("stats", {})
	if isinstance(obj, dict):
		return Stats.model_validate(obj)
	return Stats()


def save_stats(st: Stats) -> None:
	raw = _load_raw()
	raw["stats"] = st.model_dump()
	_add_defaults_if_missing(raw)
	_save_raw(raw)


def _add_defaults_if_missing(raw: Dict[str, Any]) -> None:
	if "settings" not in raw or not isinstance(raw["settings"], dict):
		raw["settings"] = Settings().model_dump()
	if "stats" not in raw or not isinstance(raw["stats"], dict):
		raw["stats"] = Stats().model_dump()
