SR = 44100

import io
from typing import Any, cast
import numpy as np
import numpy.typing as npt
import soundfile as sf


def tone(freq: float, dur: float, waveform: str = "sine") -> npt.NDArray[np.float32]:
	"""Generate a single tone with a simple attack/release envelope.

	Args:
		freq: Frequency in Hz
		dur: Duration in seconds
		waveform: One of {"sine","triangle","saw"}
	"""
	t = np.linspace(0.0, dur, int(SR * dur), endpoint=False, dtype=np.float32)
	omega = 2.0 * np.pi * freq
	if waveform == "sine":
		x = np.sin(omega * t).astype(np.float32)
	elif waveform == "triangle":
		# 2/pi * arcsin(sin)
		x = ((2.0 / np.pi) * np.arcsin(np.sin(omega * t))).astype(np.float32)
	else:
		# sawtooth via fractional part formula
		phase = (freq * t).astype(np.float32)
		x = (2.0 * (phase - np.floor(phase + 0.5))).astype(np.float32)

	# ADSR-lite: 5ms attack, 50ms release
	attack = int(0.005 * SR)
	release = int(0.050 * SR)
	env = np.ones_like(x, dtype=np.float32)
	if attack > 0:
		env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
	if release > 0:
		env[-release:] = np.linspace(1.0, 0.0, release, endpoint=False, dtype=np.float32)

	y = (x * env).astype(np.float32)
	return cast(npt.NDArray[np.float32], y)


def melodic(f1: float, f2: float, gap: float = 0.10, dur: float = 0.60, waveform: str = "sine") -> npt.NDArray[np.float32]:
	gap_samples = int(SR * gap)
	n_gap = np.zeros(gap_samples, dtype=np.float32)
	return np.concatenate([tone(f1, dur, waveform), n_gap, tone(f2, dur, waveform)])


def harmonic(f1: float, f2: float, dur: float = 1.0, waveform: str = "sine") -> npt.NDArray[np.float32]:
	x = tone(f1, dur, waveform) + tone(f2, dur, waveform)
	max_abs = float(np.max(np.abs(x))) if x.size else 1.0
	if max_abs > 0.0:
		x = (x / max_abs).astype(np.float32)
	return cast(npt.NDArray[np.float32], x)


def wav_bytes(x: npt.NDArray[np.float32]) -> bytes:
	buf = io.BytesIO()
	sf.write(buf, x, SR, format="WAV")
	return buf.getvalue()
