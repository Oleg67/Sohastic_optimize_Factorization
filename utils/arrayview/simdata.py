import numpy as np
import datetime

from .. import timestamp, YEAR, DAY, HOUR
from ..accumarray import step_indices, unpack, accum, accum_np


def fill_simdata(av, events=30, runs_per_race=10, runners=None, seed=None, random_factors=0, build_derivates=False):
    rnd = _get_rnd(seed)
    av_size = events * runs_per_race + 1
    av.event_id = np.concatenate(([0], np.repeat(np.arange(events, dtype=int), runs_per_race) + 1))
    courses = np.maximum(np.minimum(int(events / 3), 50), 1)
    av.course = av.event_id % courses + 1  # Course should not be 0

    obstacle_sel = list('FHC') * 5 + ['']
    obstacle_choice = av.event_id % len(obstacle_sel)
    av.obstacle = np.array(obstacle_sel, dtype='S1')[obstacle_choice]

    going_sel = ['HARD', 'GOOD', 'SOFT', 'AW', 'HEAVY']
    going_choice = av.event_id % len(going_sel)
    av.going = np.array(going_sel, dtype='S5')[going_choice]

    av.weight = av.event_id % 5 + 50 + rnd.randint(0, 10, av_size)
    av.run_id = np.arange(av_size)

    av.prize = accum(av.event_id, (rnd.rand(av_size) * 500).astype(int))

    if runners is None:
        runners = events
    av.runner_id = create_ids(av.event_id, runners, racewise_unique=(runs_per_race <= events), rnd=rnd)
    av.jockey = create_ids(av.event_id, np.maximum(int(runners / 30), 2 * runs_per_race), rnd=rnd)
    av.trainer = create_ids(av.event_id, np.maximum(int(runners / 15), 2 * runs_per_race), racewise_unique=False, rnd=rnd)
    nsires = runners // 10
    av.sire = av.runner_id % nsires + np.max(av.runner_id) + 1
    av.fstats_sire = av.runner_id % nsires + np.max(av.runner_id) + 1
    av.dam = av.runner_id % (runners // 2) + np.max(av.runner_id) + 1 + nsires

    today = float(timestamp(datetime.date.today().strftime("%Y-%m-%d")))
    av.date_of_birth = today - 10 * YEAR + YEAR * (av.runner_id % 5)

    av.bfid = np.arange(av_size, dtype=int) + 4000000
    av.bdid = np.arange(av_size, dtype=int) + 3000000

    times = np.arange(events) * DAY + today - 4 * YEAR + 10 * HOUR + (np.arange(events) % 8) * HOUR
    # Start times should not be assumed to be linearly increasing for some tests
    rnd.shuffle(times)
    av.distance = 1000 + obstacle_choice * 100 + rnd.randint(0, 10) * 100
    av.win_time = av.distance / 14 * (1 + going_choice / 3) * (1 + obstacle_choice / len(obstacle_sel)) + rnd.rand(av_size)
    av.speed = av.distance / av.win_time
    av.start_time = np.concatenate(([0], np.repeat(times, runs_per_race)))

    av.bsp = rnd.rand(av_size) * 999 + 1

    av.create_col('step1probs')
    for i in xrange(1, events * runs_per_race + 1, runs_per_race):
        chance = 1 / rnd.rand(runs_per_race)
        chance /= np.sum(chance)
        av.step1probs[i:i + runs_per_race] = chance

    if not random_factors:
        av.result = np.concatenate(([0], ((av.run_id[1:] - 1) % runs_per_race) + 1))
    else:
        # Create artificial factors and derive results from them
        factors = rnd.randn(random_factors, av_size)
        for n in xrange(random_factors):
            col = "factor%s" % n
            av.create_col(col)
            av[col] = factors[n, :]

        # "orginal/true" coefficients
        coef_org = rnd.randn(random_factors, 1)

        # Computes horse strength and winning probabilities
        strength = np.dot(coef_org.transpose(), factors).reshape(-1)
        expV = np.exp(strength)
        expsum = accum(av.event_id, expV)
        av.true_prob = expV / unpack(av.event_id, expsum)

        # Draw results and winners
        av.result = draw_results(av.event_id, av.true_prob, rnd=rnd)
    av.draw = ((av.result + av.event_id % 10) % runs_per_race) + 1
    av.beaten_length = np.where(av.result - 1, ((av.result - 1) * 2 + rnd.rand(len(av))) * 2, 0)
    av.sex[av.runner_id % 2 == 1] = 'H'
    av.sex[av.runner_id % 2 == 0] = 'M'

    if build_derivates:
        av.speed_cogd = rnd.rand(av_size) * 5 + 10
        av.norm_speed = rnd.randn(len(av))
        av.norm_speed_cogd = rnd.randn(len(av))
        av.norm_result = rnd.rand(len(av)) - 0.5

    # No operation should have forced the av to resize
    assert len(av) == av_size
    return av


def fill_ts_from_av(ts, av, steps=10, timestep=120, seed=None, nonrunner_chance=0.03):
    rnd = _get_rnd(seed)
    event_ids = np.unique(av.event_id)
    idx = 0
    bankroll = 20000 + rnd.rand() * 5000
    for event_id in event_ids[1:]:
        rows = np.where(av.event_id == event_id)[0]
        nruns = len(rows)
        starttime = av.start_time[rows[0]]
        chance = 1 / rnd.rand(nruns)
        chance = _shake(chance, intensity=0, rnd=rnd)
        timestamp = starttime - (steps - 1) * timestep - 1
        nonrunner = np.zeros_like(rows, dtype=bool)
        bvol = _shake(chance, rnd=rnd) * 10
        lvol = _shake(chance, rnd=rnd) * 10

        for _ in xrange(steps):
            sl = slice(idx, idx + nruns)
            ts.grow(idx + nruns)
            ts.back_price[sl] = np.round(1 / _shake(chance, intensity=0, norm=1.05, rnd=rnd), 2)
            ts.lay_price[sl] = np.round(1 / _shake(chance, intensity=0, norm=0.95, rnd=rnd), 2)
            ts.back_volume[sl] = bvol
            ts.lay_volume[sl] = lvol
            ts.reduction_factor[sl] = _shake(chance, intensity=0.1, rnd=rnd)
            ts.timestamp[sl] = timestamp
            ts.event_id[sl] = event_id
            ts.run_id[sl] = av.run_id[rows]
            ts.in_play_delay[sl] = timestamp >= starttime
            ts.nonrunner[sl] = nonrunner
            ts.bankroll[sl] = bankroll

            bvol += _shake(chance, rnd=rnd) * 30
            lvol += _shake(chance, rnd=rnd) * 30
            bankroll += rnd.rand() * 300 + 10
            nonrunner |= rnd.rand(nruns) > 1 - nonrunner_chance
            chance[nonrunner] = np.nan
            chance = _shake(chance, intensity=0.05, rnd=rnd)
            timestamp += timestep
            idx += nruns

    ts.sort('timestamp')
    return ts


def draw_results(strata, prob, rnd=np.random):
    '''Draws results from probabilities'''
    strata_sorted = strata.copy()
    if np.any(np.diff(strata) < 0):  # step_count(strata) != len(np.unique(strata)):
        strata_sorted = np.sort(strata)
    sortidx = np.argsort(strata, kind='mergesort')
    prob_sorted = prob[sortidx]

    indices = step_indices(strata_sorted)
    results = np.zeros(len(prob_sorted), dtype=int)
    for i in xrange(len(indices) - 1):
        idx = np.arange(indices[i], indices[i + 1])
        p = prob_sorted[idx].astype(np.float64)
        p /= np.sum(p)
        for result in xrange(1, len(idx) + 1):
            winnumber = rnd.choice(np.arange(len(idx)), p=p)
            results[idx[winnumber]] = result
            if result < len(idx):
                p[winnumber] = 0
                p /= np.sum(p)

    invsortidx = np.argsort(sortidx, kind='mergesort')
    return results[invsortidx]


def draw_winners(strata, prob, rnd=np.random):
    '''Draws winners from probabilities'''
    strata_sorted = strata.copy()
    if np.any(np.diff(strata) < 0):  # step_count(strata) != len(np.unique(strata)):
        strata_sorted = np.sort(strata)
    sortidx = np.argsort(strata, kind='mergesort')
    prob_sorted = prob[sortidx]

    indices = step_indices(strata_sorted)
    winners = np.zeros(len(prob_sorted), dtype=bool)
    for i in xrange(len(indices) - 1):
        idx = np.arange(indices[i], indices[i + 1])
        winnumber = rnd.choice(np.arange(len(idx)), p=prob_sorted[idx])
        winners[idx[winnumber]] = True

    invsortidx = np.argsort(sortidx, kind='mergesort')
    return winners[invsortidx]


def create_ids(strata, nRunners, racewise_unique=True, rnd=np.random):
    nData = len(strata)
    rid = rnd.randint(1, nRunners, nData)
    if not racewise_unique:
        return rid
    alldifferent = lambda x:len(x) == len(np.unique(x))
    badraces = np.ones(nData, dtype=np.bool_)
    count = 0
    while np.any(badraces):
        count += 1
        rid[badraces] = rnd.randint(1, nRunners, np.sum(badraces))
        badraces = unpack(strata, accum_np(strata, rid, func=alldifferent) == 0)
        if count > 1000:
            raise ArithmeticError('ID creation did not succeed.')

    # TODO: Zero should become a valid value one day
    rid[0] = 0
    return rid


def _get_rnd(seed):
    if seed is None:
        return np.random
    else:
        return np.random.RandomState(seed)


def _shake(vals, intensity=0.1, norm=1, rnd=np.random):
    ret = vals + rnd.rand(len(vals)) * intensity
    return np.minimum(np.maximum(np.round(norm * ret / np.nansum(ret), 2), 0.001), 0.99)
