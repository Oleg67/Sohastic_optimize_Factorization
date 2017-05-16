import pytest
import os
import numpy as np

from ... import settings
from ..arraycontainer import ArrayContainer, _tuple_decode, _tuple_encode
from ..arrayview import ArrayView


def _validname(name):
    if name and all(c.isalnum() or c == '_' for c in name) and name[0] != '_':
        return True
    return False

column_names = ['asdf', 'name_name', 'mean(bla,blub)', '!"$%&/()=?',
                'asdf_2342__,.$%$%_', '_asdf', 'bla____']

@pytest.mark.parametrize("col", column_names)
def test_tuple_encoding(col):
    if col.endswith('__'):
        pytest.raises(AssertionError, _tuple_encode, col)
        assert _validname(col), "Encoding marker set on invalid field"
        return

    if _validname(col):
        assert _tuple_encode(col) == col, "Encoding was unnecessarily applied"
    else:
        encoded = _tuple_encode(col)
        assert encoded != col, "No encoding was applied"
        assert encoded.endswith('__'), "No encoding marker was set"
        assert _validname(encoded), "Encoding invalid for tuple fields"


@pytest.mark.parametrize("col", column_names[:-2])
def test_tuple_decode(col):
    encoded = _tuple_encode(col)
    assert _tuple_decode(encoded) == col


@pytest.fixture(params=[ArrayContainer, ArrayView], ids=lambda x: x.__name__)
def ac(request):
    ac = request.param(allocation=100)
    ac.grow(100)
    for col in ('foo', 'bar', 'mean(bla, blub)'):
        ac.create_col(col)
        ac[col][:] = np.arange(100)
    for col in ('flt', 'blb'):
        ac.create_col(col)
        ac[col][:] = np.arange(100, dtype=float)
        ac[col][::3] = np.nan
    extra_data = dict(testdata='notanarray', testint=20, testarr=np.arange(10, dtype=int))
    ac.extra.update(extra_data)
    return ac


def test_len(ac):
    assert len(ac) == 100
    assert ac.allocated == 100


def test_resize_allocation(ac):
    org_len, org_alloc = len(ac), ac.allocated
    ac.resize_allocation(2 * org_alloc)
    assert len(ac) == org_len
    assert ac.allocated == 2 * org_alloc
    for col in ac:
        assert len(ac[col]) == org_len
        assert len(ac._cols[col]) == ac.allocated


def test_grow(ac):
    newlen = len(ac) * 2
    ac.grow(newlen)
    assert len(ac) == newlen
    for col in ac:
        assert len(ac[col]) == newlen
    assert ac.allocated > newlen


def test_memsize(ac):
    assert ac.memsize() == sum(c.nbytes for c in ac._cols.itervalues())


def test_slice(ac):
    newac = ac[10:100:2]
    assert len(newac) == 45
    for col in ac:
        np.testing.assert_array_equal(newac[col], ac[col][10:100:2])


def test_prohibit_erroneous_writes(ac):
    with pytest.raises(AttributeError):
        ac.newattr = None
    assert 'newattr' not in ac._cols
    with pytest.raises(TypeError):
        ac['newindicator'] = 'notanarray'
    assert 'notanarray' not in ac._cols


def test_equal(ac):
    clone = ac.copy()
    assert ac == clone
    assert ac is not clone
    clone.create_col('ddd')
    assert ac != clone

    clone = ac.copy()
    # Allocation is a hidden detail, so it should not impact comparison
    clone.resize_allocation(2 * clone.allocated)
    assert ac == clone

    clone.grow(2 * len(clone))
    assert ac != clone

    clone = ac.copy()
    clone.foo[2] = 222
    assert ac != clone


def test_copy(ac):
    clone = ac.copy()
    for col in ac:
        assert ac._cols[col] is not clone._cols[col]
        assert len(ac[col]) > 0
        np.testing.assert_array_equal(ac[col], clone[col])


def test_shrink(ac):
    sel = np.s_[::2]
    org_len = len(ac)
    org_foo = ac.foo
    ac.shrink(sel)
    assert len(ac) == org_len // 2
    np.testing.assert_array_equal(ac.foo, org_foo[sel])


def test_set_value(ac):
    assert ac.get_value('foo', 10) != 3
    ac.set_value('foo', 10, 3)
    assert ac.foo[10] == 3
    assert ac.get_value('foo', 10) == 3


def test_hide_none_storage(ac):
    ac.set_value('bar', 0, None)
    assert np.isnan(ac.bar[0])
    assert ac.get_value('bar', 0) is None
    assert ac.get_value('bar', 1) is not None


def test_setrange(ac):
    sel = np.s_[:]
    dummydata = np.arange(len(ac), dtype=float)
    ac.create_col('asdf')
    ac.asdf[sel] = dummydata
    assert hasattr(ac, 'asdf')
    np.testing.assert_array_equal(ac.asdf, dummydata)


def test_automatic_reallocation(ac):
    realloc = len(ac) * 2
    ac.set_value('foo', realloc, realloc)
    assert ac.foo[realloc] == realloc
    assert ac.allocated > realloc


def test_flush(ac):
    org_alloc = ac.allocated
    ac.extra.bla = 10
    ac.flush()
    assert len(ac) == 0
    assert not len(ac._cols)
    assert not len(ac.extra)
    assert ac.allocated == org_alloc


def test_namedtuple(ac):
    nt_ac = ac.namedtuple()
    assert isinstance(nt_ac, tuple)
    for field in ac._cols:
        np.testing.assert_equal(getattr(nt_ac, _tuple_encode(field)), ac[field])


def test_from_tuple(ac):
    ac_clone = type(ac).from_tuple(ac.namedtuple())
    assert ac_clone == ac


@pytest.mark.parametrize("filetype", ['npz', 'bcolz'])
def test_purge(tmpdir, ac, filetype):
    dump_path = str(tmpdir.join(type(ac).__name__.lower() + '_dummy.av.' + filetype))
    ac.set_path(dump_path)

    assert not os.path.exists(ac.path)
    assert not os.path.isfile(ac.path_extra)
    try:
        ac.dump(ac.path)
        assert os.path.exists(ac.path)
        assert os.path.isfile(ac.path_extra)
    finally:
        ac.purge()

    assert not os.path.exists(ac.path)
    assert not os.path.isfile(ac.path_extra)


@pytest.mark.parametrize("filetype", ['npz', 'bcolz'])
def test_overwrite(tmpdir, ac, filetype):
    dump_path = str(tmpdir.join(type(ac).__name__.lower() + '_dummy.av.' + filetype))
    ac.set_path(dump_path)

    try:
        ac.dump()
        assert os.path.exists(ac.path)
        ac.dump()
    finally:
        ac.purge()


@pytest.mark.parametrize("filetype", ['npz', 'bcolz'])
def test_dump_restore(tmpdir, ac, filetype):
    dump_path = str(tmpdir.join(type(ac).__name__.lower() + '_dummy.av.' + filetype))
    ac.set_path(dump_path)
    assert len(ac.extra) > 2

    try:
        ac.dump()
        clone = type(ac).from_file(ac.path)
        assert os.path.exists(ac.path), "No container file dumped"
        assert os.path.isfile(ac.path_extra), "No container extra file dumped"
    finally:
        ac.purge()

    assert len(ac) == len(clone)
    assert clone.path == ac.path
    for field in ac:
        assert field in clone
    assert ac == clone
    for key in sorted(ac.extra):
        assert key in clone.extra
        if isinstance(ac.extra[key], np.ndarray):
            np.testing.assert_array_equal(ac.extra[key], clone.extra[key])
        else:
            assert ac.extra[key] == clone.extra[key]
