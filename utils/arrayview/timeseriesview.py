import numpy as np
import numba as nb

from .. import get_logger, timestamp, intnan
from .arraycontainer import ArrayContainer

logger = get_logger(__package__)


class TimeseriesView(ArrayContainer):
    _spread_av_tmp = None
    _col_types = dict(timestamp='float64', event_id='int64', run_id='int64', back_volume='float32', lay_volume='float32',
                      back_price='float32', lay_price='float32', price='float32', volume='float32', reduction_factor='float32',
                      in_play_delay='int8', slice_idx='int8', bankroll='float32', nonrunner='bool')
    # TODO: Throw out slice_idx, and make this a lazy attribute
    fields = sorted(_col_types)

    @classmethod
    def from_db(cls, db, ts_start=0, ts_end=np.inf, account=None, verbose=False):
        if isinstance(ts_start, str):
            ts_start = float(timestamp(ts_start))
        if isinstance(ts_end, str):
            ts_end = float(timestamp(ts_end))

        event_iter = db.root['Event'].unique.values(min=(ts_start, 0), max=(ts_end, 1000))
        ts = cls.prepared(allocation=len(event_iter) * 1000)

        for ev in db.safe_iter(event_iter, chunk_len=1000):
            # Get all timeseries for one event together
            ev_tss = [db.root['Timeseries'].unique.get((run.id, 'BF')) for run in ev.runs]
            misses = sum(1 for (run_ts, run) in zip(ev_tss, ev.runs) if not run_ts and not run.non_runner)
            if misses:
                if verbose:
                    logger.info("Event with %s missing timeseries found, skipping: %s", misses, ev.short_description)
                continue

            try:
                full_ts_len = max(len(run_ts) for run_ts in ev_tss if run_ts)
            except ValueError:
                continue
            if full_ts_len < 10:
                # Skip bogus entries
                continue

            # Determine a timestamp, which will also be used for nonrunner values
            for run_ts in ev_tss:
                if run_ts and len(run_ts.local_timestamp) == full_ts_len:
                    ref_timestamp = run_ts.local_timestamp
                    break
            else:
                logger.warning("No reference timestamp found: %s, ts_len: %s", ev.short_description, full_ts_len)
                continue

            for (run_ts, run) in zip(ev_tss, ev.runs):
                start_idx = len(ts)
                end_idx = start_idx + full_ts_len
                ts.grow(end_idx)
                ts.run_id[start_idx:end_idx] = run.id
                ts.timestamp[start_idx:end_idx] = ref_timestamp
                ts.event_id[start_idx:end_idx] = run.event.id

                if not run_ts:
                    # Do the quickest shortcut for missing values
                    ts.nonrunner[start_idx:end_idx] = True
                    continue

                # If len(run_ts) == full_ts_len: No nonrunner flags will be set
                ts.nonrunner[start_idx + len(run_ts):end_idx] = True
                # logger.info("Extracting: %s, len %s, idx %s:%s", run_ts.run.description, len(run_ts), start_idx, end_idx)
                for colname in ('back_volume', 'lay_volume', 'back_price', 'lay_price',
                                'reduction_factor', 'in_play_delay'):
                    tsv_col = getattr(ts, colname)
                    col = getattr(run_ts, colname)
                    # logger.info("Current col %s (%s) %s", colname, type(col).__name__, col)
                    try:
                        tsv_col[start_idx:end_idx][:len(col)] = col
                    except ValueError:
                        tsv_col[start_idx:end_idx][:len(col)] = [val if isinstance(val, (float, int)) else np.nan for val in col]

        # It's already sorted by event and always the same order of runners,
        # so now have everything sorted by timestamp/event
        ts.sort('timestamp')

        if account and np.any(np.isnan(ts.bankroll)):
            ts.bankroll_from_bets(db, account, verbose=verbose)
        return ts

    def bankroll_from_bets(self, db, account, preserve_time=10, verbose=False):
        """ Calculate bankroll from previous bets """
        # Preserve time reduced to 10 seconds compared to original version, due to
        # higher frequency of bankroll updates (maximum every 10 seconds).
        if verbose:
            logger.info("Restoring bankroll data from bets")

        ts_start = np.min(self.timestamp)
        ts_end = np.max(self.timestamp)

        bets = db.bets_by_time(ts_start, ts_end, account=account, min_id=45000)
        if not bets:
            return

        if verbose:
            logger.info("Found %s bets to restore bankroll", len(bets))

        last_timestamp = -preserve_time
        bets.sort(key=lambda b: b.created_timestamp)
        bankroll = np.full(len(self), np.nan, dtype=float)
        for b in bets:
            bankroll[(self.timestamp > last_timestamp + preserve_time) & (self.timestamp <= b.created_timestamp + preserve_time)] = b.log_bankroll
            last_timestamp = b.created_timestamp
        last_reduction = b.amount if b.amount > 0 else np.abs(b.liability)
        bankroll[self.timestamp > last_timestamp + preserve_time] = b.log_bankroll - last_reduction

        # Select nan or empty values
        mask = np.logical_or(np.isnan(self.bankroll), np.logical_not(self.bankroll))
        self.bankroll[mask] = bankroll[mask]

    def slices(self):
        indices = np.where(np.ediff1d(self.strata(), to_begin=[1], to_end=[1]))[0]
        for idx in xrange(len(indices) - 1):
            yield slice(indices[idx], indices[idx + 1])

    @staticmethod
    @nb.njit(cache=True)
    def _strata(event_id, timestamp):
        strata = np.zeros(len(event_id), dtype=np.int64)
        strata[0] = 0
        cnt = 0
        for i in range(1, len(event_id)):
            if (timestamp[i] != timestamp[i - 1]) | (event_id[i] != event_id[i - 1]):
                cnt += 1
            strata[i] = cnt
        return strata

    def strata(self):
        assert len(self) == len(self.event_id) == len(self.timestamp)
        assert (np.diff(self.timestamp) >= 0).all(), "Timestamps not sorted"
        return self._strata(self.event_id, self.timestamp)


    @staticmethod
    @nb.njit(cache=True)
    def _lastvalid(field, run_id, last_valid_lookup):
        last_valid_idx = np.zeros(len(field), dtype=np.int64)
        for i, v in enumerate(field):
            if not np.isnan(v):
                last_valid_lookup[run_id[i]] = i
            last_valid_idx[i] = last_valid_lookup[run_id[i]]
        return last_valid_idx

    def lastvalid(self, field):
        """ Get the row indices of the last valid entries for different data fields """
        assert np.min(self.run_id) >= 0
        uruns = np.unique(self.run_id)
        last_valid_lookup = np.full(intnan.nanmax(uruns) + 1, intnan.INTNAN64, dtype=int)
        return self._lastvalid(self[field], self.run_id, last_valid_lookup)

    def sanity_check(self, tolerance=0.1):
        assert np.any(self.nonrunner), "Not a single non-runner found"
        assert not np.all(self.nonrunner), "All runners are non-runner"
        assert np.all(np.diff(self.timestamp) >= 0), "Timeseries not ordered by time"
        valid = ~self.nonrunner
        assert np.all((self.back_price[valid] >= 1.01) & (self.back_price[valid] <= 1000)), "Invalid back price found"
        assert np.all((self.lay_price[valid] >= 1.01) & (self.lay_price[valid] <= 1000)), "Invalid lay price found"

        allowed = len(self) * tolerance
        def check_nans(column, valid=valid, allowed=allowed):
            return np.count_nonzero(np.isnan(column[valid])) < allowed

        assert check_nans(self.back_price), "Many NAN back prices for valid runners found"
        assert check_nans(self.lay_price), "Many NAN lay prices for valid runners found"
        assert check_nans(self.back_volume), "Many NAN back volumes for valid runners found"
        assert check_nans(self.lay_price), "Many NAN lay volumes for valid runners found"

    @classmethod
    def dummy_from_av(cls, av, steps=10, timestep=120, seed=None, nonrunner_chance=0.03):
        from .simdata import fill_ts_from_av
        ts_size = len(av) * steps
        ts = cls.prepared(allocation=int(ts_size * cls.overallocate))
        fill_ts_from_av(ts, av, steps, timestep, seed, nonrunner_chance)
        return ts
