from eartrainer.theory import SEMITONES, A4_MIDI, A4_FREQ, midi_to_freq, interval_to_pair, note_to_midi, settings_range_to_midi


def test_midi_to_freq_a4():
	assert midi_to_freq(A4_MIDI) == A4_FREQ


def test_interval_to_pair_ascending_descending():
	root = 60
	m1, m2 = interval_to_pair(root, "M3", "ascending")
	assert (m1, m2) == (60, 64)
	m1, m2 = interval_to_pair(root, "M3", "descending")
	assert (m1, m2) == (60, 56)


def test_note_to_midi_and_range():
	assert note_to_midi("C3") == 48
	lo, hi = settings_range_to_midi(("C3","A4"))
	assert lo == 48 and hi == 69
