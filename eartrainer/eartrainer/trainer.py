from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from .models import AnswerRecord, Question, Settings, Stats
from .theory import SEMITONES, interval_names, interval_to_pair, midi_pair_to_freqs, pick_root, pick_root_in_bounds, settings_range_to_midi


class AdaptiveState:
	def __init__(self, intervals: List[str], window: int = 50) -> None:
		self.window = window
		self.history: Dict[str, Deque[bool]] = {name: deque(maxlen=window) for name in intervals}
		self.recent_misses: Deque[str] = deque(maxlen=3)

	def seed_from_stats(self, stats: Stats) -> None:
		for name, st_i in stats.by_interval.items():
			if name not in self.history:
				self.history[name] = deque(maxlen=self.window)
			# Approximate last-window accuracy by filling deque proportionally
			n = min(self.window, max(1, st_i.seen))
			k = int(round((st_i.correct / st_i.seen) * n)) if st_i.seen else 0
			self.history[name].clear()
			self.history[name].extend([True] * k + [False] * (n - k))

	def accuracy(self, name: str) -> float:
		h = self.history.get(name)
		if not h or len(h) == 0:
			return 0.0
		return sum(1 for x in h if x) / float(len(h))

	def update(self, interval: str, correct: bool) -> None:
		self.history.setdefault(interval, deque(maxlen=self.window)).append(bool(correct))
		if not correct:
			self.recent_misses.append(interval)

	def weights(self, intervals: List[str], min_floor: float = 0.05) -> List[float]:
		# Base score: 1 - accuracy
		scores = []
		for name in intervals:
			base = 1.0 - self.accuracy(name)
			# Recent-mistake boost
			if name in self.recent_misses:
				base += 0.15
			scores.append(max(0.0, base))
		# Softmax
		exps = np.exp(scores - np.max(scores) if scores else 0.0)
		w = (exps / np.sum(exps)).tolist() if np.sum(exps) > 0 else [1.0 / max(1, len(intervals))] * len(intervals)
		# Apply min floor
		if len(w) > 0:
			w = np.maximum(w, min_floor)
			w = (w / np.sum(w)).tolist()
		return [float(x) for x in w]


def choose_interval(intervals: List[str], weights: List[float]) -> str:
	idx = int(np.random.choice(len(intervals), p=weights))
	return intervals[idx]


def distractors(correct: str, k: int = 3) -> List[str]:
	# Pick nearest semitone intervals to correct as distractors
	names = interval_names()
	names = [n for n in names if n != correct]
	target = SEMITONES[correct]
	names.sort(key=lambda n: abs(SEMITONES[n] - target))
	candidates = names[: max(1, k + 2)]
	# Occasionally insert a far distractor for variety
	if len(names) > 0 and random.random() < 0.2:
		far = names[-1]
		if far not in candidates:
			candidates[-1] = far
	return random.sample(candidates, k=min(k, len(candidates)))


def make_question(settings: Settings, state: AdaptiveState) -> Question:
	intervals = settings.intervals
	w = state.weights(intervals)
	name = choose_interval(intervals, w)
	min_midi, max_midi = settings_range_to_midi(settings.range)
	root = pick_root_in_bounds(min_midi, max_midi, name, settings.mode)
	m1, m2 = interval_to_pair(root, name, settings.mode)
	options = [name] + distractors(name, k=3)
	random.shuffle(options)
	return Question(
		interval=name,
		options=options,
		answer=name,
		root_midi=root,
		pair_midi=(m1, m2),
		mode=settings.mode,
	)


def score_answer(q: Question, chosen: str, stats: Stats, state: AdaptiveState) -> AnswerRecord:
	is_correct = chosen == q.answer
	# Update stats
	if q.interval not in stats.by_interval:
		from .models import IntervalStats
		stats.by_interval[q.interval] = IntervalStats()
	stats.by_interval[q.interval].seen += 1
	if is_correct:
		stats.by_interval[q.interval].correct += 1
	else:
		conf = stats.confusion.setdefault(q.interval, {})
		conf[chosen] = conf.get(chosen, 0) + 1
	# Update adaptive state
	state.update(q.interval, is_correct)
	return AnswerRecord(interval=q.interval, chosen=chosen, correct=is_correct)
