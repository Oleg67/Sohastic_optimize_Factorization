"""
ArrayView groups all database information excluding timeseries data
as numpy arrays. It makes best effort to provide actualized data,
but can't always promise that.
"""

import time
import os
from collections import namedtuple
from functools import partial
from itertools import chain
import numpy as np

from .. import get_logger, list_get, chunks_sized, intnan, dispdots
from ..accumarray import accum_sort
from ..database import Run, get_key_part
from .arraycontainer import ArrayContainer, _tuple_encode


logger = get_logger(__package__)

fieldnames_event = 'start_time course distance going obstacle race_class win_time prize'.split()
fieldnames_runner = 'date_of_birth sex dam sire'.split()
fieldnames_run_lists = ['morning_prob_slice', 'morning_est', 'morning_prob', 'thb_est']
fieldnames_run = sorted(Run._properties - Run._key_properties_set - set(fieldnames_run_lists))
fieldnames_tipsters = ['napstats_tipssum']
fieldnames_ids = ['event_id', 'run_id', 'runner_id']

fieldname_lists = fieldnames_ids, fieldnames_event, fieldnames_runner, fieldnames_run, fieldnames_run_lists[1:], fieldnames_tipsters
fieldname_list_all = list(chain.from_iterable(fieldname_lists))
fieldname_list_provided = list(chain(fieldnames_ids, fieldnames_event, fieldnames_runner, fieldnames_run))

fieldtypes = dict(obstacle='S1', sex='S1', race_class='S2', going='S5', condition='S30', name='S20', result='int8', draw='int8',
                  morning_est_times='float64', morning_prob='float32', morning_prob_times='float64', thb_est='float32',
                  tipsters='int8', napstats_tipssum='int8', ones='int8')
for fieldname in ['jockey', 'trainer', 'course', 'dam', 'sire', 'strata', 'bfid', 'bdid'] + fieldnames_ids:
    fieldtypes[fieldname] = 'int64'


def get_col_type(col):
    try:
        return fieldtypes[col]
    except KeyError:
        if col.endswith('_int') or col.endswith('_id'):
            return 'int64'
        elif col.endswith('_double'):
            return 'float64'
        elif col.endswith('_string'):
            return 'S20'
        elif '_nfp_' in col or col.endswith('_wnr') or col.endswith('_frac'):
            return 'float32'
        elif col.endswith('time'):
            return 'float64'
        for item in ('winnum', 'pref_num', 'numraces', 'numruns', 'samenum'):
            if item in col:
                return 'int32'
    return 'float32'



class ArrayView(ArrayContainer):
    """ ArrayContainer extended with database interface """
    fields = fieldname_list_all
    _row_lut_cache = None
    _default_values = dict(napstats_tipssum=0)

    def _tainted(self, extra=True):
        self._row_lut_cache = None
        if extra:
            self.extra.clear()

    def col_type(self, col):
        return np.dtype(get_col_type(col))

    def sync(self, db, lookback=0, chunk_size=10000):
        """ Synchronize db data with own arrays. Return the updated ids if any. """
        if not db.loaded:
            raise RuntimeError("Cannot open closed database")
        for field in fieldname_list_all:
            if field not in self:
                self.create_col(field)

        maxrun = max(np.max(self.run_id) if len(self) else 0, 0) if len(self) else 0
        new_runs = db.root.Run.id.values(min=maxrun)
        updated_run_ids = set()
        updated_event_ids = set()
        if new_runs:
            cbprint = partial(dispdots, dotcnt=chunk_size)
            for runs_chunk in chunks_sized(db.safe_iter(new_runs, chunk_len=chunk_size, callback=cbprint), chunk_size):
                updated_run_ids.update(self.update_runs(runs_chunk, db=db))

        if len(self) and lookback:
            # Check for recently added values required for indicator calculation
            for ev in db.root.Event.unique.values(min=(time.time() - lookback, 0)):
                run_ids = ev.run_ids
                updated_event_ids.add(ev.id)
                if any(rid not in updated_run_ids for rid in run_ids):
                    self.update_runs(ev.runs, db=db)
                    updated_run_ids.update(run_ids)

        return np.array(sorted(updated_run_ids))

    def update_runs(self, runs, db=None):
        """
        Update AV fields with the given list of runs. runs needs to be a proper list, not just an iterator,
        so that it can be iterated several times for evaluating newly created rows in advance.
        """
        updated_run_ids = []
        run_ids = np.array([r.id for r in runs])
        row_ids = self.row_lookup(run_ids)
        new_runs = np.sum(row_ids < 0)
        row_ids[row_ids < 0] = np.arange(new_runs, dtype=int) + len(self)
        self.grow(len(self) + new_runs)

        set_val = self._set_value_raw
        for row_id, run in zip(row_ids, runs):
            event = run.event
            runner = run.runner
            set_val('event_id', row_id, event.id)
            set_val('run_id', row_id, run.id)
            set_val('runner_id', row_id, runner.id)
            for field in fieldnames_event:
                set_val(field, row_id, get_key_part(getattr(event, field)))
            for field in fieldnames_runner:
                set_val(field, row_id, get_key_part(getattr(runner, field)))
            for field in fieldnames_run:
                set_val(field, row_id, get_key_part(getattr(run, field)))
            for field in fieldnames_run_lists[1:]:
                set_val(field, row_id, get_key_part(getattr(run, field).last))

            if db is not None:
                # TODO: This is not really efficient. Maybe tips should be a simple list within the Run object?
                tips = [tip for tip in db.tips_by_run(run) if tip.service == 'NAP']
                set_val('napstats_tipssum', row_id, len(tips))
            updated_run_ids.append(run.id)
        self._tainted()
        return updated_run_ids

    def _guess_len(self):
        valid = np.where(~intnan.isnan(self._cols.run_id))[0]
        if len(valid):
            return np.max(valid) + 1
        return 0

    def load(self, path=None, require_all=False):
        super(ArrayView, self).load(path=path)
        if require_all:
            missing = [field for field in self.fields if field not in self._cols]
            if missing:
                logger.info("Could not load data: Required fields not found: %s", ','.join(missing))
                return

    def set_value(self, field, index, value):
        value = get_key_part(value)
        ArrayContainer.set_value(self, field, index, value)
        if field == 'run_id':
            # Invalidate cache
            self._tainted()

    def row_lookup(self, run_id):
        lut = self.row_lut
        try:
            return lut[run_id]
        except (IndexError, ValueError):
            if isinstance(run_id, int):
                return intnan.INTNAN64
            else:
                return np.fromiter((list_get(lut, rid, intnan.INTNAN64) for rid in run_id), dtype=int)

    @property
    def row_lut(self):
        if self._row_lut_cache is not None:
            return self._row_lut_cache
        lut = self._row_lut_cache = self._row_lut()
        return lut

    def _row_lut(self):
        """ Create a lookup table, to convert run_ids into row_ids for lookup av[lut[run_id]] """
        if not len(self):
            lut = np.ones(0, dtype=int)
        elif not 'run_id' in self:
            return
        else:
            maxid = np.max(self.run_id)
            if maxid < 0:
                lut = np.full(len(self), intnan.INTNAN64, dtype=int)
            else:
                lut = np.full(maxid + 1, intnan.INTNAN64, dtype=int)
        valid = self.run_id >= 0
        run_ids = self.run_id[valid]
        lut[run_ids] = np.arange(len(self.run_id), dtype=int)[valid]
        return lut

    def trim(self):
        """ Get rid of invalid rows """
        sel = (self.run_id >= 0) & (self.event_id >= 0)
        if not np.all(sel):
            self.shrink(sel)

    def shuffled(self, sortidx=None, reverse=False, return_index=False):
        if sortidx is None:
            sortidx = np.random.permutation(len(self))
            # Here magic happens. Makes sure that the sort index does not
            # shuffle entries with equal start time
            sortidx = accum_sort(self.start_time[sortidx], sortidx)

        if reverse:
            sortidx = np.argsort(sortidx, kind='mergesort')
        ret = self[sortidx]
        if return_index:
            return ret, sortidx
        return ret

    def namedtuple(self, fields=None):
        fields = list(fields) if fields else sorted(self.keys())
        new_dict = dict((_tuple_encode(k), self[k]) for k in fields)
        fields = [_tuple_encode(f) for f in fields]
        if self.row_lut is not None:
            fields.append('row_lut')
            new_dict['row_lut'] = self.row_lut
        return namedtuple(type(self).__name__ + 'Tuple', fields)(**new_dict)

    @classmethod
    def from_db(cls, db, path=None, npz=False):
        suffix = '.av.npz' if npz else '.av.bcolz'
        path = path or db.path and db.path[:-5] + suffix
        av = cls.prepared(allocation=int(len(db.root.Run.id) * 1.05), path=path)
        av.sync(db, lookback=0)
        return av

    @classmethod
    def from_db_cached(cls, db, npz=False):
        if not db.path or not db.path.endswith('.zodb'):
            raise ValueError('Invalid database path: %s' % db.path)
        for ending in ('bcolz', 'npz'):
            path = db.path[:-5] + '.av.' + ending
            if os.path.exists(path):
                av = cls.from_file(path)
                if npz and not path.endswith('npz'):
                    av.set_path(path[:-5] + 'npz')
                return av
        else:
            raise OSError('No stored AV file found')

    @classmethod
    def dummy(cls, events=30, runs_per_race=10, runners=None, seed=None, random_factors=0,
              sim_start=None, build_derivates=False):
        from .simdata import fill_simdata
        av_size = events * runs_per_race + 1
        av = cls.prepared(allocation=int(av_size * cls.overallocate))
        fill_simdata(av, events, runs_per_race, runners, seed, random_factors,
                     sim_start=sim_start, build_derivates=build_derivates)
        return av
