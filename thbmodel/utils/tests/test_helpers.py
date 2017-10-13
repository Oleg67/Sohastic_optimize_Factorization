import pytest
import time

from ..helpers import chunks_sized, find_items
from ..extensions import cached, cached_ttl

@pytest.mark.parametrize("chunk_size", [-1, 0, 1, 3, 7, 10, 20])
def test_chunks_sized(chunk_size):
    vals = list(range(20))
    if chunk_size < 1:
        with pytest.raises(ValueError):
            chunks_sized(vals, chunk_size).next()
    else:
        total_size = 0
        for chunk in chunks_sized(vals, chunk_size):
            assert isinstance(chunk, list)
            assert 0 < len(chunk) <= chunk_size
            total_size += len(chunk)
        assert total_size == len(vals)


find_int_input1 = [1, 2, 3, [5, 6, 6, {'a': 12, 'b': 55, 'c': 'asdf', 'f': {'d': 'c', 'e': 33}}, 6, 4.4, object(), 7, range(3)]]
find_int_input2 = [1, 2, [5, 6, xrange(3)]]

find_int_results = [
    (None, find_int_input1, [1, 2, 3, 5, 6, 6, 12, 55, 33, 6, 7, 0, 1, 2]),
    (1, find_int_input1, [1, 2, 3]),
    (2, find_int_input1, [1, 2, 3, 5, 6, 6, 6, 7]),
    (3, find_int_input1, [1, 2, 3, 5, 6, 6, 12, 55, 6, 7, 0, 1, 2]),
    (4, find_int_input1, [1, 2, 3, 5, 6, 6, 12, 55, 33, 6, 7, 0, 1, 2]),
    (None, find_int_input2, [1, 2, 5, 6, 0, 1, 2]),
    (1, find_int_input2, [1, 2]),
    (2, find_int_input2, [1, 2, 5, 6]),
    (3, find_int_input2, [1, 2, 5, 6, 0, 1, 2]),
]
@pytest.mark.parametrize(["recursion_depth", "inputs", "ref"], find_int_results)
def test_find_items_int(recursion_depth, inputs, ref):
    assert list(find_items(inputs, lambda x: isinstance(x, int), recursion_depth=recursion_depth)) == ref


def test_find_items_str():
    string_test = ['asdf', 3, 5, 6, dict(a='as', b='tt', c=1, d=dict(a=1, c='dd')), 'tt']
    string_result = ['asdf', 'as', 'tt', 'dd', 'tt']
    assert list(find_items(string_test, lambda x: isinstance(x, str))) == string_result


def test_cached():
    class CachedClass(object):
        step = 0
        @cached
        def cached_prop(self):
            self.step += 1
            return self.step

    c = CachedClass()
    ref = [c.cached_prop for _ in range(3)]
    assert ref == [1, 1, 1]


cached_ttl_inputs = [
    (1.5, [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
    (0.6, [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
    (None, [1] * 12),  # Never invalidate
]
@pytest.mark.parametrize(["multiplier", "ref"], cached_ttl_inputs)
def test_cached_ttl(monkeypatch, multiplier, ref):
    timebase = 0.01
    class CachedClass(object):
        step = 0
        @cached_ttl(timebase)
        def cached_ttl_prop(self):
            self.step += 1
            return self.step

    curtime = 100
    def mockreturn():
        return curtime
    monkeypatch.setattr(time, 'time', mockreturn)

    cached_obj = CachedClass()
    resps = []
    for _ in range(4):
        for _ in range(3):
            resps.append(cached_obj.cached_ttl_prop)
        if multiplier:
            curtime += multiplier * timebase
    assert resps == ref
    assert cached_obj._v_cached_ttl_prop_cached == ref[-1]
