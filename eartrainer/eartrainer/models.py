from __future__ import annotations

from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field


Mode = Literal["ascending", "descending", "harmonic"]
Waveform = Literal["sine", "triangle", "saw", "piano"]


class Settings(BaseModel):
	intervals: List[str] = Field(
		default=["m2","M2","m3","M3","P4","TT","P5","m6","M6","m7","M7","P8"]
	)
	mode: Mode = Field(default="ascending")
	range: Tuple[str, str] = Field(default=("C3", "A4"))
	waveform: Waveform = Field(default="piano")
	session_len: int = Field(default=20, ge=1, le=200)
	volume: float = Field(default=0.9, ge=0.0, le=1.0)


class IntervalStats(BaseModel):
	seen: int = 0
	correct: int = 0


class Stats(BaseModel):
	by_interval: Dict[str, IntervalStats] = Field(default_factory=dict)
	confusion: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class Question(BaseModel):
	interval: str
	options: List[str]
	answer: str
	root_midi: int
	pair_midi: Tuple[int, int]
	mode: Mode


class AnswerRecord(BaseModel):
	interval: str
	chosen: str
	correct: bool


class SessionResult(BaseModel):
	settings: Settings
	answers: List[AnswerRecord]
	final_stats: Stats
