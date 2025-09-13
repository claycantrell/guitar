import streamlit as st
import pandas as pd
import altair as alt
from typing import Any, Dict
import base64

from eartrainer.audio import harmonic as synth_harmonic, melodic as synth_melodic, wav_bytes
from eartrainer.models import AnswerRecord, Settings, Mode, Waveform, Stats
from eartrainer.storage import load_settings, load_stats, save_settings, save_stats
from eartrainer.theory import midi_pair_to_freqs
from eartrainer.trainer import AdaptiveState, make_question, score_answer


st.set_page_config(page_title="Ear Trainer", page_icon=None, layout="centered")


def get_state() -> Any:
	if "settings" not in st.session_state:
		st.session_state.settings = load_settings()
	if "stats" not in st.session_state:
		st.session_state.stats = load_stats()
	if "adaptive" not in st.session_state:
		st.session_state.adaptive = AdaptiveState(st.session_state.settings.intervals)
		# Seed from existing stats so adaptation works from the first question
		st.session_state.adaptive.seed_from_stats(st.session_state.stats)
	if "current_question" not in st.session_state:
		st.session_state.current_question = None
	# removed replays_left
	if "seen" not in st.session_state:
		st.session_state.seen = 0
	if "correct" not in st.session_state:
		st.session_state.correct = 0
	if "history" not in st.session_state:
		st.session_state.history = []
	if "feedback" not in st.session_state:
		st.session_state.feedback = None  # {"correct": bool, "text": str}
	if "trigger_autoplay" not in st.session_state:
		st.session_state.trigger_autoplay = False
	if "play_version" not in st.session_state:
		st.session_state.play_version = 0
	if "clear_audio" not in st.session_state:
		st.session_state.clear_audio = False
	if "session_stats" not in st.session_state:
		st.session_state.session_stats = Stats()
	return st.session_state


def sidebar_controls(s: Settings) -> Settings:
	st.sidebar.header("Settings")
	intervals = st.sidebar.multiselect(
		"Intervals",
		options=["m2","M2","m3","M3","P4","TT","P5","m6","M6","m7","M7","P8"],
		default=s.intervals,
	)
	mode_str = st.sidebar.selectbox("Mode", ["ascending","descending","harmonic"], index=["ascending","descending","harmonic"].index(s.mode))
	range_val = st.sidebar.select_slider("Range (root)", options=["C3","D3","E3","F3","G3","A3","B3","C4","D4","E4","F4","G4","A4"], value=s.range)
	waveform_str = st.sidebar.selectbox("Waveform", ["sine","triangle","saw"], index=["sine","triangle","saw"].index(s.waveform))
	session_len = st.sidebar.slider("Session length", min_value=5, max_value=100, value=s.session_len, step=1)
	volume = st.sidebar.slider("Volume", min_value=0.0, max_value=1.0, value=s.volume, step=0.05)

	mode: Mode = mode_str  # type: ignore[assignment]
	waveform: Waveform = waveform_str  # type: ignore[assignment]
	new_s = Settings(intervals=intervals or s.intervals, mode=mode, range=range_val, waveform=waveform, session_len=session_len, volume=volume)
	if new_s != s:
		save_settings(new_s)
	return new_s


@st.cache_data(show_spinner=False)
def _cached_audio_bytes(f1: float, f2: float, mode: str, waveform: str, salt: int) -> bytes:
	if mode == "harmonic":
		x = synth_harmonic(f1, f2, dur=1.0, waveform=waveform)
	else:
		x = synth_melodic(f1, f2, gap=0.10, dur=0.60, waveform=waveform)
	return wav_bytes(x)


def main() -> None:
	state = get_state()
	s = sidebar_controls(state.settings)
	state.settings = s

	st.title("Interval Ear Trainer")

	# Next only
	if st.button("Next", use_container_width=True):
		state.current_question = make_question(state.settings, state.adaptive, state.current_question)
		state.feedback = None
		state.trigger_autoplay = True
		state.play_version += 1

	# Feedback indicator (green/red)
	if state.feedback is not None:
		if state.feedback.get("correct"):
			st.success(state.feedback.get("text", "Correct!"))
		else:
			st.error(state.feedback.get("text", "Incorrect"))

	# Playback if we have a question
	if state.current_question is not None:
		player = st.empty()
		# Phase 1: clear any existing audio element, then schedule autoplay
		if state.clear_audio:
			player.empty()
			state.clear_audio = False
			state.trigger_autoplay = True
			state.play_version += 1
			st.rerun()

		m1, m2 = state.current_question.pair_midi
		f1, f2 = midi_pair_to_freqs(m1, m2)
		bytes_ = _cached_audio_bytes(f1, f2, state.current_question.mode, state.settings.waveform, state.play_version)
		if state.trigger_autoplay:
			player.empty()
			player.audio(bytes_, format="audio/wav", autoplay=True)
			state.trigger_autoplay = False
		else:
			player.audio(bytes_, format="audio/wav", autoplay=False)

		st.subheader("Choose the interval")
		btn_cols = st.columns(2)
		options = state.current_question.options
		clicked = None
		for idx, label in enumerate(options):
			with btn_cols[idx % 2]:
				if st.button(label, key=f"opt-{label}", use_container_width=True):
					clicked = label

		if clicked is not None:
			record = score_answer(state.current_question, clicked, state.stats, state.adaptive)
			state.seen += 1
			if record.correct:
				state.correct += 1
			state.history.append(record)
			save_stats(state.stats)
			# Update session-scoped stats (no adaptive update here)
			if state.current_question.interval not in state.session_stats.by_interval:
				from eartrainer.models import IntervalStats
				state.session_stats.by_interval[state.current_question.interval] = IntervalStats()
			state.session_stats.by_interval[state.current_question.interval].seen += 1
			if record.correct:
				state.session_stats.by_interval[state.current_question.interval].correct += 1
			else:
				conf = state.session_stats.confusion.setdefault(state.current_question.interval, {})
				conf[clicked] = conf.get(clicked, 0) + 1
			# Set feedback indicator (green/red)
			if record.correct:
				state.feedback = {"correct": True, "text": "Correct!"}
				st.success("Correct!")
			else:
				msg = f"Incorrect — correct was {state.current_question.answer}"
				state.feedback = {"correct": False, "text": msg}
				st.error(msg)
			if state.seen >= state.settings.session_len:
				st.success("Session complete. See results below.")
			else:
				# Do not auto-advance or autoplay on answer; wait for Next
				pass

	# Score and end session
	st.markdown("---")
	st.write(f"Score: {state.correct} / {state.seen}")
	if st.button("End Session"):
		# Reset session counters and session-scoped results but keep persistent stats
		state.current_question = None
		state.seen = 0
		state.correct = 0
		state.history = []
		state.feedback = None
		state.trigger_autoplay = False
		state.clear_audio = False
		state.session_stats = Stats()

	# Results table (simple for MVP)
	if state.session_stats.by_interval:
		st.subheader("Results by interval")
		rows = []
		for name, st_i in state.session_stats.by_interval.items():
			acc = (st_i.correct / st_i.seen) if st_i.seen else 0.0
			rows.append({"interval": name, "seen": st_i.seen, "correct": st_i.correct, "accuracy": round(acc, 3)})
		st.dataframe(rows, hide_index=True)

	if state.session_stats.confusion:
		st.subheader("Confusion heatmap")
		# Flatten confusion dict to long-form DataFrame
		data = []
		for truth, d in state.session_stats.confusion.items():
			for chosen, count in d.items():
				data.append({"truth": truth, "chosen": chosen, "count": count})
		df = pd.DataFrame(data)
		chart = alt.Chart(df).mark_rect().encode(
			x=alt.X("chosen:N", sort=None),
			y=alt.Y("truth:N", sort=None),
			color=alt.Color("count:Q", scale=alt.Scale(scheme="oranges")),
			tooltip=["truth","chosen","count"],
		).properties(width=400, height=300)
		st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
	main()
