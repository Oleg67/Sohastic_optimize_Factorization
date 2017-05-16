import sys
import functools
import types
import string
from collections import Iterable
from contextlib import contextmanager

def sign(x):
    return cmp(x, 0)


def object_by_id(id_):
    import gc
    for obj in gc.get_objects():
        if id(obj) == id_:
            return obj
    raise Exception("Not found")


def get_first(itemlist, comparator, default=NotImplemented):
    for x in itemlist:
        if comparator(x):
            return x
    if default == NotImplemented:
        raise ValueError("No matching item in list")
    else:
        return default


def uniquify(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


def list_get(l, index, default=None):
    try:
        return l[index]
    except IndexError:
        return default


def broadcast(func, iterables=(list, tuple)):
    """ Decorator for optionally applying a function on every item of an iterable """
    from functools import wraps

    @wraps(func)
    def wrapper(inputs):
        if isinstance(inputs, iterables):
            return [func(x) for x in inputs]
        else:
            return func(inputs)
    return wrapper


def chunks(iterable, key):
    """ Splits iterables in chunks with the same key. See also itertools.groupby """
    buf = None
    for item in iterable:
        if not buf:
            buf = [item]
            compare_me = key(item)
        elif compare_me == key(item):
            buf.append(item)
        else:
            yield buf
            buf = [item]
            compare_me = key(item)
    if buf:
        yield buf


def chunks_sized(iterable, chunk_size):
    if chunk_size < 1:
        raise ValueError("Invalid chunk size")
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def iter_except(func, exception, first=None):
    """ Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like __builtin__.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.

    Examples:
        bsddbiter = iter_except(db.next, bsddb.error, db.first)
        heapiter = iter_except(functools.partial(heappop, h), IndexError)
        dictiter = iter_except(d.popitem, KeyError)
        dequeiter = iter_except(d.popleft, IndexError)
        queueiter = iter_except(q.get_nowait, Queue.Empty)
        setiter = iter_except(s.pop, KeyError)

    """
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass


def raiser(exception_type, *args):
    """
    Kleiner Helfer, um direkt aus x-if-y-else-z-Konstrukten Exceptions raisen zu koennen.
    Beispiel: z = lambda x: x if x > 5 else raiser(ValidationError, 'x < 5!')
    """
    __tracebackhide__ = True  # For py.test
    if not args:
        raise exception_type()
    elif len(args) == 1:
        raise exception_type(args[0])
    else:
        raise exception_type(args[0] % args[1:])


@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass


def debug_on(*exceptions):
    if not exceptions:
        exceptions = (AssertionError,)
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                import traceback
                import pdb
                traceback.print_exc()
                pdb.post_mortem(sys.exc_info()[2])
        return wrapper
    return decorator


def mklist(data):
    if data is None:
        return []
    if type(data) not in (list, tuple):
        return [data]
    return data


def mkset(data, frozen=True):
    if data is None:
        return set()
    if type(data) not in (list, tuple, set, frozenset):
        data = [data]
    if type(data) in (list, tuple):
        if frozen:
            data = frozenset(data)
        else:
            data = set(data)
    return data


def transpose_cols(arrs):
    return zip(*arrs)


def reverse_cols(arrs):
    """ Tauschen der Reihenfolge von Spalten """
    rev = list(zip(*arrs))
    rev.reverse()
    return zip(*rev)


def find_all(needle, haystack, pos=0):
    """ Wrapper fuer str.find, um alle Fundstellen nacheinander durchzugehen """
    while True:
        pos = haystack.find(needle, pos)
        if pos == -1:
            return
        else:
            yield pos
            pos += 1


def find_items(item, key, recursion_depth=None):
    """"
    Walk recoursively through all given lists and dictionaries and pick instances,
    which evaluate key(item) to True.
    """
    if recursion_depth is not None:
        if recursion_depth < 0:
            raise StopIteration
        else:
            recursion_depth -= 1
    if key(item):
        yield item
    elif isinstance(item, Iterable) and not isinstance(item, basestring):
        if hasattr(item, 'values'):
            item = item.values()
        for x in item:
            for y in find_items(x, key, recursion_depth=recursion_depth):
                yield y
    else:
        raise StopIteration


# http://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_equal(lst):
    # Best solution for short lists or if no frequent breaks are expected
    return not lst or lst.count(lst[0]) == len(lst)


def all_equal_iterated(lst):
    # Iterates and breaks on first unequal. Optimal solution for all other cases
    try:
        iterator = iter(lst)
        first = next(iterator)
        return all(first == rest for rest in iterator)
    except StopIteration:
        return True


def dispdots(i, dotcnt):
    """ Displays some I-am-doing-something dots """
    if i > 0:
        if i % dotcnt == 0:
            print '.',
            sys.stdout.flush()
        if i % (10 * dotcnt) == 0:
            print i
