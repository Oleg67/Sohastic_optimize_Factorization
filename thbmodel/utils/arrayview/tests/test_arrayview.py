import pytest
import numpy as np

from ...intnan import INTNAN64

from ..arrayview import ArrayView, get_col_type
from ..timeseriesview import TimeseriesView

runners = 10
events = 20


@pytest.fixture(scope='module')
def av():
    return ArrayView.dummy(runs_per_race=runners, events=events)


def test_get_col_types():
    for col in 'event_id run_id runner_id jockey trainer course bfid asdf_int'.split():
        assert get_col_type(col) == 'int64'
    for col in 'sex race_class name'.split():
        assert get_col_type(col).startswith('S')
    for col in 'bsp date_of_birth prize'.split():
        assert get_col_type(col) == 'float32'
    assert get_col_type('napstats_tipssum') == 'int8'


def test_col_type(av):
    assert av.jockey.dtype == np.int
    assert av.trainer.dtype == np.int
    assert av.event_id.dtype == np.int
    assert av.run_id.dtype == np.int
    assert av.runner_id.dtype == np.int
    assert av.bsp.dtype == np.float32
    assert av.napstats_tipssum.dtype == np.int8


@pytest.mark.parametrize("field", ['runner_id', 'event_id', 'trainer', 'result', 'napstats_tipssum'])
def test_set_value(av, field):
    org_val = av.get_value(field, 10)
    new_val = (org_val + 1) * 2
    av.set_value(field, 10, new_val)
    assert getattr(av, field)[10] == new_val
    assert av.get_value(field, 10) == new_val


def test_hide_none_storage_int(av):
    av.set_value('jockey', 10, None)
    assert av.jockey[10] == INTNAN64
    assert av.get_value('jockey', 10) is None
    assert av.get_value('jockey', 1) is not None


def test_shuffle(av):
    shuf = av.shuffled()
    assert np.mean(av.run_id == shuf.run_id) < 0.1
    assert sorted(shuf.run_id) == sorted(av.run_id)


def test_trim():
    av = ArrayView.dummy()
    org_len = len(av)
    av.run_id[-10:] = INTNAN64
    invalids = np.sum(av.run_id == INTNAN64)
    av.trim()
    assert len(av) == org_len - invalids


def test_row_lut_negative():
    av = ArrayView.prepared()
    av.grow(1)
    lut = av._row_lut()
    assert len(lut) == len(av)
    assert np.all(lut < 0)


def test_row_lookup_growing():
    av = ArrayView.prepared()
    assert len(av) == 0
    assert av.row_lookup(0) < 0
    av.grow(1)
    assert len(av) == 1
    assert av.row_lookup(1) < 0
    av.grow(2)
    assert len(av) == 2
    assert av.row_lookup(2) < 0


def test_row_lookup(av):
    shuf = av.shuffled()
    lst = np.arange(len(av), dtype=int)
    for i in range(0, len(av), 5):
        assert shuf.row_lookup(i) == np.where(shuf.run_id == i)[0]
    assert np.all(shuf.run_id[shuf.row_lookup(lst)] == lst)


def test_namedtuple(av):
    nt_av = av.namedtuple()
    assert nt_av.row_lut is not None
    np.testing.assert_equal(nt_av.row_lut, av.row_lut)
    for field in av._cols:
        np.testing.assert_equal(getattr(nt_av, field), av[field])


@pytest.mark.slow
@pytest.mark.parametrize("filetype", ["npz", "bcolz"])
def test_large_dump(tmpdir, filetype):
    dump_path = str(tmpdir.join('arrayview_dummy.av.' + filetype))
    av = ArrayView.dummy(events=100000, random_factors=20)
    av.set_path(dump_path)

    new_fields = ['sum(whatever)',
                  '-mult(something)',
                  'oper(one, oper(two, three))',
                  'mult(something x anything, var(blubber, blabber))',
                  'grmpf(rgmbl, rampf(hurz, furz))',
                  'whatever(you, want(from x me, gnarf))',
                  'more(foooooood, gieeeeve)',
                  'why(do, iiii(really, need))',
                  'to(do(this, very), stupid(testing, forr))',
                  'this(fuckin(segfault, bug), inn(thiss, shitttttttty))',
                  'bclrz(library, spoils(my, day))']

    for field in new_fields:
        av.create_col(field)
    av.dump()


@pytest.mark.parametrize("seed", [10, 1000, 10000])
def test_tsdummy(seed, events=300):
    av = ArrayView.dummy(events=events, seed=seed)
    ts = TimeseriesView.dummy_from_av(av, seed=seed)
    ts.sanity_check()

    av_ref = ArrayView.dummy(events=events, seed=seed)
    ts_ref = TimeseriesView.dummy_from_av(av, seed=seed)
    assert av == av_ref, "AV differs despite same seed"
    assert ts == ts_ref, "TS differs despite same seed"
