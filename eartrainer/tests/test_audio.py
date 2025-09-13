import numpy as np

from eartrainer.audio import SR, harmonic, melodic, tone


def test_tone_length_and_dtype():
	dur = 0.5
	x = tone(440.0, dur, waveform="sine")
	assert isinstance(x, np.ndarray)
	assert x.dtype == np.float32
	assert len(x) == int(SR * dur)


def test_melodic_gap_and_concat():
	dur = 0.6
	gap = 0.1
	x = melodic(440.0, 660.0, gap=gap, dur=dur, waveform="sine")
	expected = int(SR * dur) * 2 + int(SR * gap)
	assert len(x) == expected


def test_harmonic_normalization():
	x = harmonic(440.0, 660.0, dur=1.0, waveform="sine")
	assert np.max(np.abs(x)) <= 1.0 + 1e-6
