import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import requests
import mido
import io
import numpy as np
import soundfile as sf

SF2_DIR = Path.home() / ".eartrainer" / "sf2"
SF2_DIR.mkdir(parents=True, exist_ok=True)
# Small, decent upright piano soundfont (FreePats project), ~6MB
DEFAULT_SF2_URL = "https://freepats.zenvoid.org/Piano/UprightPianoKW/UprightPianoKW-small-SF2-20190703.7z"
# Fallback full SF2 (27MB) if we later add extraction; for now we prefer small one if we can fetch a direct sf2.
# Many FreePats archives are .7z; to avoid external extractors, we could host/cache a direct .sf2.
# For MVP, attempt another small direct SF2: bright variant (~6MB)
ALT_SF2_URL = "https://freepats.zenvoid.org/Piano/UprightPianoKW/UprightPianoKW-small-SF2-20190703.7z"

# To avoid 7z extraction requirement, allow user-supplied local SF2 via env
ENV_SF2_PATH = os.environ.get("EARTRAINER_SF2_PATH")


def _which(cmd: str) -> bool:
	return shutil.which(cmd) is not None


def is_piano_available() -> bool:
	# Needs fluidsynth binary OR user-supplied SF2 plus fluidsynth
	if not _which("fluidsynth"):
		return False
	p = get_sf2_path()
	return p.exists()


def get_sf2_path() -> Path:
	# Prefer env override
	if ENV_SF2_PATH:
		return Path(ENV_SF2_PATH)
	return SF2_DIR / "UprightPianoKW-small.sf2"


def ensure_sf2() -> None:
	p = get_sf2_path()
	if p.exists():
		return
	# We need a direct .sf2 link; FreePats small are 7z; so for MVP, short-circuit: if env variable missing, do nothing.
	# User can place an SF2 at ~/.eartrainer/sf2/UprightPianoKW-small.sf2 or set EARTRAINER_SF2_PATH.
	# To provide a better UX, we can later host a small sf2 mirror.
	return


def _write_midi(temp_mid: Path, notes: Tuple[int, int], mode: str, volume: float) -> None:
	mid = mido.MidiFile()
	trk = mido.MidiTrack()
	mid.tracks.append(trk)
	# Set a tempo so that 1 beat = 0.6s (100 BPM)
	trk.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(100), time=0))
	# Channel 0, program 0 (Acoustic Grand)
	trk.append(mido.Message('program_change', program=0, time=0))
	m1, m2 = notes
	# Scale velocity by volume (60..120)
	vel = max(1, min(127, int(60 + 60 * volume)))
	# Ticks per beat is 480 by default
	TPB = mid.ticks_per_beat  # 480
	# Durations at 100 BPM: 1 beat = 0.6s
	NOTE_TICKS = TPB  # 0.6s
	GAP_TICKS = int(TPB * (0.1 / 0.6))  # ~80 ticks
	HARM_TICKS = int(TPB * (1.0 / 0.6))  # ~800 ticks
	if mode == "harmonic":
		trk.append(mido.Message('note_on', note=m1, velocity=vel, time=0))
		trk.append(mido.Message('note_on', note=m2, velocity=vel, time=0))
		trk.append(mido.Message('note_off', note=m1, velocity=0, time=HARM_TICKS))
		trk.append(mido.Message('note_off', note=m2, velocity=0, time=0))
	else:
		trk.append(mido.Message('note_on', note=m1, velocity=vel, time=0))
		trk.append(mido.Message('note_off', note=m1, velocity=0, time=NOTE_TICKS))
		# gap before second
		trk.append(mido.Message('note_on', note=m2, velocity=vel, time=GAP_TICKS))
		trk.append(mido.Message('note_off', note=m2, velocity=0, time=NOTE_TICKS))
	mid.save(temp_mid.as_posix())


def _trim_to_duration(wav_bytes: bytes, seconds: float, sr_target: int = 44100) -> bytes:
	data, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')
	# Ensure mono by mixing down if stereo
	if data.ndim == 2:
		data = data.mean(axis=1)
	# Resample if needed would be overkill; fluidsynth default is 44100, we force -r 44100
	target_samples = int(sr_target * seconds)
	data = data[:target_samples]
	buf = io.BytesIO()
	sf.write(buf, data, sr_target, format='WAV')
	return buf.getvalue()


def render_piano_bytes(m1: int, m2: int, mode: str, volume: float) -> bytes:
	ensure_sf2()
	sf2 = get_sf2_path()
	if not sf2.exists():
		raise RuntimeError("SF2 not available. Set EARTRAINER_SF2_PATH to a valid .sf2 or place one in ~/.eartrainer/sf2/UprightPianoKW-small.sf2")
	if not _which("fluidsynth"):
		raise RuntimeError("fluidsynth not found. Install with: brew install fluidsynth")
	with tempfile.TemporaryDirectory() as td:
		dirp = Path(td)
		midp = dirp / "tmp.mid"
		wavp = dirp / "out.wav"
		_write_midi(midp, (m1, m2), mode, volume)
		cmd = [
			"fluidsynth",
			"-g", "1.2",          # increase synth gain a bit
			"-R", "0",             # disable reverb to remove tail
			"-C", "0",             # disable chorus
			"-r", "44100",         # ensure 44.1k sample rate
			"-F", wavp.as_posix(),
			sf2.as_posix(),
			midp.as_posix(),
		]
		proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		if proc.returncode != 0 or not wavp.exists():
			raise RuntimeError(f"fluidsynth failed: {proc.stderr.decode(errors='ignore')}")
		raw = wavp.read_bytes()
		target_seconds = 1.3 if mode == "harmonic" else 1.6
		return _trim_to_duration(raw, target_seconds, 44100) 