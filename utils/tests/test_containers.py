import pytest
import time
import string

from ..containers import AttrDict, DefaultNoneAttrDict, NoneTraversal

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
