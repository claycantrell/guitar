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

# Direct download link for FluidR3Mono_GM soundfont (~14MB compressed SF3)
# SF3 is a compressed SF2 format that FluidSynth natively supports
DEFAULT_SF2_URL = "https://github.com/musescore/MuseScore/raw/2.3.2/share/sound/FluidR3Mono_GM.sf3"

# To avoid 7z extraction requirement, allow user-supplied local SF2 via env
ENV_SF2_PATH = os.environ.get("EARTRAINER_SF2_PATH")

# General MIDI program numbers for different instruments
INSTRUMENT_PROGRAMS = {
	"piano": 0,  # Acoustic Grand Piano
	"acoustic_guitar": 24,  # Acoustic Guitar (nylon)
	"electric_guitar_clean": 27,  # Electric Guitar (clean)
	"electric_guitar_jazz": 26,  # Electric Guitar (jazz)
}


def _which(cmd: str) -> bool:
	return shutil.which(cmd) is not None


def is_soundfont_available() -> bool:
	"""Check if FluidSynth and a soundfont are available for rendering instruments."""
	if not _which("fluidsynth"):
		return False
	ensure_sf2()  # Attempt to download if not present
	p = get_sf2_path()
	return p.exists()

# Backward compatibility alias
def is_piano_available() -> bool:
	return is_soundfont_available()


def get_sf2_path(instrument: str = "piano") -> Path:
	# Prefer env override
	if ENV_SF2_PATH:
		return Path(ENV_SF2_PATH)
	
	# For non-piano instruments, use FluidR3Mono_GM (has all GM instruments)
	if instrument != "piano":
		fluid_path = SF2_DIR / "FluidR3Mono_GM.sf3"
		if fluid_path.exists():
			return fluid_path
	
	# Check for YDP-GrandPiano (piano only, high quality)
	ydp_path = SF2_DIR / "YDP-GrandPiano-SF2-20160804" / "YDP-GrandPiano-20160804.sf2"
	if ydp_path.exists():
		return ydp_path
	
	# Fall back to FluidR3Mono (if downloaded for other instruments)
	fluid_path = SF2_DIR / "FluidR3Mono_GM.sf3"
	if fluid_path.exists():
		return fluid_path
	
	# Otherwise use the default path (for auto-download or symlink)
	return SF2_DIR / "UprightPianoKW-small.sf2"


def ensure_sf2() -> None:
	p = get_sf2_path()
	if p.exists():
		return
	# If env variable is set but file doesn't exist, don't auto-download
	if ENV_SF2_PATH:
		return
	# Auto-download FluidR3Mono_GM soundfont to default location
	try:
		response = requests.get(DEFAULT_SF2_URL, timeout=30)
		response.raise_for_status()
		p.write_bytes(response.content)
	except Exception as e:
		# Silent fail - user will see error message later if they try to use piano
		pass


def _write_midi(temp_mid: Path, notes: Tuple[int, int], mode: str, volume: float, program: int = 0) -> None:
	mid = mido.MidiFile()
	trk = mido.MidiTrack()
	mid.tracks.append(trk)
	# Set a tempo so that 1 beat = 0.6s (100 BPM)
	trk.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(100), time=0))
	# Channel 0, program number for selected instrument
	trk.append(mido.Message('program_change', program=program, time=0))
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


def render_instrument_bytes(m1: int, m2: int, mode: str, volume: float, instrument: str = "piano") -> bytes:
	"""Render notes using FluidSynth with the specified instrument from the soundfont."""
	ensure_sf2()
	sf2 = get_sf2_path(instrument)  # Pass instrument to get correct soundfont
	if not sf2.exists():
		raise RuntimeError("SF2 not available. Set EARTRAINER_SF2_PATH to a valid .sf2 or place one in ~/.eartrainer/sf2/UprightPianoKW-small.sf2")
	if not _which("fluidsynth"):
		raise RuntimeError("fluidsynth not found. Install with: brew install fluidsynth")
	
	# Get the MIDI program number for the instrument
	program = INSTRUMENT_PROGRAMS.get(instrument, 0)
	
	with tempfile.TemporaryDirectory() as td:
		dirp = Path(td)
		midp = dirp / "tmp.mid"
		wavp = dirp / "out.wav"
		_write_midi(midp, (m1, m2), mode, volume, program)
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


def render_piano_bytes(m1: int, m2: int, mode: str, volume: float) -> bytes:
	"""Backward compatibility wrapper for piano rendering."""
	return render_instrument_bytes(m1, m2, mode, volume, "piano") 