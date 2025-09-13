from eartrainer.models import Settings, Stats
from eartrainer.trainer import AdaptiveState, make_question


def test_make_question_structure():
	settings = Settings()
	state = AdaptiveState(settings.intervals)
	q = make_question(settings, state)
	assert q.interval in settings.intervals
	assert len(q.options) == 4
	assert q.answer in q.options
	m1, m2 = q.pair_midi
	assert isinstance(m1, int) and isinstance(m2, int)


def test_adaptive_weights_boosts_weak_intervals():
	settings = Settings(intervals=["m2","P5"])
	state = AdaptiveState(settings.intervals)
	# Seed state to make P5 look mastered and m2 weak
	from eartrainer.models import IntervalStats, Stats
	stats = Stats(by_interval={"m2": IntervalStats(seen=10, correct=2), "P5": IntervalStats(seen=10, correct=9)})
	state.seed_from_stats(stats)
	w = state.weights(settings.intervals)
	# Weight for m2 should be higher than for P5
	w_m2 = w[settings.intervals.index("m2")]
	w_p5 = w[settings.intervals.index("P5")]
	assert w_m2 > w_p5
