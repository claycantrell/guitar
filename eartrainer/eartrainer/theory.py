from typing import List, Tuple

SEMITONES = {
	"m2": 1,
	"M2": 2,
	"m3": 3,
	"M3": 4,
	"P4": 5,
	"TT": 6,
	"P5": 7,
	"m6": 8,
	"M6": 9,
	"m7": 10,
	"M7": 11,
	"P8": 12,
}

A4_MIDI = 69
A4_FREQ = 440.0

NOTE_TO_MIDI = {
	"C3": 48,
	"D3": 50,
	"E3": 52,
	"F3": 53,
	"G3": 55,
	"A3": 57,
	"B3": 59,
	"C4": 60,
	"D4": 62,
	"E4": 64,
	"F4": 65,
	"G4": 67,
	"A4": 69,
}


def midi_to_freq(m: int) -> float:
	return float(A4_FREQ * (2.0 ** ((m - A4_MIDI) / 12.0)))


def pick_root(mmin: int = 48, mmax: int = 69) -> int:
	import random
	return random.randint(mmin, mmax)


def interval_to_pair(root_midi: int, name: str, direction: str) -> Tuple[int, int]:
	d = SEMITONES[name]
	if direction == "descending":
		d = -d
	return root_midi, root_midi + d


def midi_pair_to_freqs(m1: int, m2: int) -> Tuple[float, float]:
	return midi_to_freq(m1), midi_to_freq(m2)


def interval_names() -> List[str]:
	return list(SEMITONES.keys())


def note_to_midi(note: str) -> int:
	return NOTE_TO_MIDI[note]


def settings_range_to_midi(range_notes: Tuple[str, str]) -> Tuple[int, int]:
	lo, hi = range_notes
	return note_to_midi(lo), note_to_midi(hi)


def pick_root_in_bounds(min_midi: int, max_midi: int, interval_name: str, mode: str) -> int:
	"""Pick a root so that the second note stays within [min_midi, max_midi]."""
	import random
	d = SEMITONES[interval_name]
	if mode == "descending":
		low = min_midi + d
		high = max_midi
	else:
		low = min_midi
		high = max_midi - d
	low = max(low, min_midi)
	high = min(high, max_midi)
	if low > high:
		# Fallback to simple picker if range too tight
		return pick_root(min_midi, max_midi)
	return random.randint(low, high)
