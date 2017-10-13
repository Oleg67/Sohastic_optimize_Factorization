import pytest
import time
import string

from ..containers import Cache, AttrDict, DefaultNoneAttrDict, NoneTraversal

dict_content = dict(a=10, b=20, c=30)

def test_attrdict_init():
    for a in (AttrDict(dict_content), AttrDict(**dict_content), AttrDict(dict_content.items())):
        for key in dict_content:
            assert key in a
            assert a[key] == dict_content[key]


def test_attrdict_content_comparison():
    a = AttrDict(dict_content)
    assert a == dict_content
    assert id(a) != id(dict_content)


def test_attrdict_copy():
    a = AttrDict(dict_content)
    b = a.copy()
    assert a is not b
    assert a == b


def test_attrdict_access():
    a = AttrDict(dict_content)
    for key in dict_content:
        assert a[key] == getattr(a, key)


def test_defaultnoneattrdict():
    a = DefaultNoneAttrDict(dict_content)
    assert 'y' not in a
    assert not a['y']
    assert not a.y


def test_nonetraversal():
    n = NoneTraversal()
    assert n == None
    assert not n != None
    assert None == n
    assert not None != n
    assert n != 0
    assert n != 1
    assert n != False
    assert bool(n) == False
    assert n.bla is n


def test_cache():
    c = Cache(10, max_age=0.1)
    bla = 'bla'
    c[bla] = bla
    assert c[bla] == bla
    time.sleep(c.max_age * 1.2)
    assert bla not in c
    with pytest.raises(KeyError):
        c.__getitem__(bla)

    c.update(dict(asdf='blabla'))
    assert c['asdf'] == 'blabla'


def test_cache_housekeeping():
    c = Cache(10, max_age=0.1)
    for val, key in enumerate(string.ascii_lowercase):
        c[key] = val
        time.sleep(c.max_age / 10.0)
    assert len(c) == c.length, "Cache grew over limitation while filling"
    c._housekeeping(intense=False)
    assert len(c) == c.length, "Non intense housekeeping already removed items without need"
    time.sleep(c.max_age * 1.2)
    c._housekeeping(intense=True)
    assert len(c) == 0, "Cache cleanup forgot some timed out items"
