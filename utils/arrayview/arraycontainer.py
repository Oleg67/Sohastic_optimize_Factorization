import os
import shutil
import base64
from collections import MutableMapping, Iterable, namedtuple
from copy import deepcopy
import numpy as np

from .. import AttrDict, get_logger, intnan
from ..math import sel_len, sleep

logger = get_logger(__package__)




def _tuple_encode(name):
    """ Convert column names into something digestable for bcolz """
    assert name, "Can not encode empty name"
    assert not name.endswith('__'), "Can not encode name ending with double underscores"
    if any(not (c.isalnum() or c == '_') for c in name) or name.startswith('_'):
        # Need to encode the column name
        return base64.b32encode(name).replace('=', '_') + '__'
    return name


def _tuple_decode(name):
    """ Restore column names from bcolz """
    if name.endswith('__'):
        return base64.b32decode(name[:-2].replace('_', '='))
    return name


class ArrayContainer(MutableMapping):
    __len = 0
    __allocation = 10000
    _cols = None  # Required for __setattr__
    extra = None  # Required for __setattr__
    path = None  # Required for __setattr__
    fields = []
    overallocate = 1.1
    _col_types = dict()
    _default_values = dict()

    def __init__(self, path=None, allocation=None):
        self._cols = AttrDict()
        self.extra = AttrDict()
        if path:
            self.set_path(path)
        if allocation:
            self.__allocation = allocation

    def set_path(self, path):
        if not (path.endswith('.av.npz') or path.endswith('.av.bcolz')):
            raise ValueError('Wrong file name ending')
        self.path = os.path.abspath(path)

    @staticmethod
    def _extra_path(path):
        return path.rsplit('.', 2)[0] + '.extra.npz'

    @property
    def path_extra(self):
        return self._extra_path(self.path) if self.path else None

    @property
    def allocated(self):
        return self.__allocation

    @property
    def loaded(self):
        return bool(self._cols) and bool(len(self))

    def col_type(self, field):
        return np.dtype(self._col_types.get(field, float))

    def default_value(self, field):
        """ Return the initial fill value for the given column name """
        try:
            return self._default_values[field]
        except KeyError:
            return intnan.NANVALS.get(self.col_type(field).char, 0)

    def _get_slice(self, sel):
        """ Create a new ArrayView instance with sliced, copied columns from self """
        new = type(self)(allocation=sel_len(sel, len(self)))
        if isinstance(sel, int):
            sel = [sel]
        for col, val in self.iteritems():
            new[col] = val[sel]
        return new

    def _get_cols(self, cols):
        new = type(self)(allocation=len(self))
        for col in cols:
            new[col] = self[col]
        return new

    def set_len(self, length):
        if not isinstance(length, int) or not 0 <= length <= self.allocated:
            raise ValueError("Invalid length")
        self.__len = length

    def resize_allocation(self, allocation, refcheck=False):
        if self.__allocation == allocation:
            return
        for field, col in self._cols.items():
            col.resize(allocation, refcheck=refcheck)
            col[self.allocated:] = self.default_value(field)
        self.__allocation = allocation
        for key in self:
            assert len(self._cols[key]) == allocation

    def grow(self, new_len, refcheck=False):
        if new_len <= len(self):
            return

        if new_len <= self.allocated:
            self.set_len(new_len)
            self._tainted()
            return

        self.resize_allocation(int(new_len * self.overallocate), refcheck=refcheck)
        self.__len = new_len
        self._tainted()

    def shrink(self, sel):
        """ Shrink the current instance to the selection """
        self.__len = self.__allocation = sel_len(sel, len(self))
        for col in self.keys():
            self._cols[col] = self._cols[col][sel].copy()
        self._tainted()

    def __getitem__(self, key):
        """ Hide overallocation of the columns """
        if isinstance(key, basestring):
            try:
                return self._cols[key][:len(self)]
            except KeyError:
                raise KeyError(key)
        elif isinstance(key, (int, slice, np.ndarray)):
            return self._get_slice(key)
        elif isinstance(key, Iterable):
            key = list(key)
            if all(isinstance(x, basestring) for x in key) and all(x in self for x in key):
                return self._get_cols(key)
            elif all(isinstance(x, int) for x in key):
                return self._get_slice(key)
        raise KeyError(key)

    def __setitem__(self, key, val):
        """ Hide overallocation of the columns """
        if isinstance(key, (slice, np.ndarray)):
            if not isinstance(val, type(self)):
                raise TypeError("Can only set other ArrayView objects as slice")
            self.grow(len(val))
            sel = key
            for key in val:
                if key in self:
                    self[key][sel] = val[key]
        elif isinstance(key, basestring):
            if not isinstance(val, (np.ndarray)):
                raise TypeError("Can only set numpy arrays as column")
            self.grow(len(val))
            if key in self._cols:
                col = self._cols[key]
            else:
                col = self.create_col(key)
            col[:len(val)] = val
        else:
            raise ValueError("Can not set key/value item combination to ArrayView")

    def __delitem__(self, key):
        del self._cols[key]

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, val):
        if hasattr(type(self), attr):
            object.__setattr__(self, attr, val)
        elif isinstance(val, (np.ndarray, list)):
            self.__setitem__(attr, val)
        else:
            raise AttributeError(attr)

    def __delattr__(self, attr):
        if attr in self._cols:
            del self._cols[attr]
        else:
            object.__delattr__(self, attr)

    def __iter__(self):
        return self._cols.__iter__()

    def __contains__(self, item):
        return item in self._cols

    def __len__(self):
        """ The arrays are overallocated, so simply taking the size does not work """
        return self.__len

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        try:
            return all(np.all(intnan.nanequal(self[col], other[col])) for col in self) and all(col in self for col in other)
        except KeyError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def create_col(self, field, length=None):
        length = length or self.allocated
        col = np.full(length, self.default_value(field), dtype=self.col_type(field))
        self._cols[field] = col
        return col

    def restore_coltypes(self, *cols):
        for field in cols or self._cols:
            col = self._cols[field]
            coltype = self.col_type(field)
            if coltype != col.dtype:
                self._cols[field] = col.astype(coltype)

    def get_value(self, field, index):
        """ Integration with objects """
        col = self[field]
        val = col[index]
        if isinstance(val, (float, np.float32, np.float64)) and np.isnan(val):
            return None
        nanval = intnan.NANVALS.get(col.dtype.char)
        if nanval == val:
            return None
        else:
            return val

    def set_value(self, field, index, value):
        self.grow(index + 1)
        self._set_value_raw(field, index, value)

    def _set_value_raw(self, field, index, value):
        col = self[field]
        value = value if value is not None else intnan.NANVALS.get(col.dtype.char, 0)
        try:
            col[index] = value
        except TypeError:
            raise TypeError("Row %s: Cannot set field %s to %s" %
                            (index, field, value))
        except IndexError:
            raise IndexError("Row %s: Cannot set field %s to %s" %
                            (index, field, value))

    def flush(self, allocation=None):
        allocation = allocation or self.allocated
        self._cols.clear()
        self.extra.clear()
        self.set_len(0)
        self._tainted()

    def copy(self):
        clone = type(self)(path=self.path, allocation=self.allocated)
        clone.set_len(len(self))
        clone.overallocate = self.overallocate
        clone._cols = deepcopy(self._cols)
        clone.extra = deepcopy(self.extra)
        return clone

    def sorted(self, by, kind='mergesort', return_index=False):  # @ReservedAssignment
        sortidx = np.argsort(self[by], kind=kind)
        ret = self[sortidx]
        if return_index:
            return ret, sortidx
        return ret

    def sort(self, by, kind='mergesort'):
        sortidx = np.argsort(self[by], kind=kind)
        for col in self._cols:
            self[col] = self[col][sortidx]
        self._tainted()
        return sortidx

    def namedtuple(self, fields=None):
        fields = fields if fields else sorted(self.keys())
        new_dict = dict((_tuple_encode(k), self[k]) for k in fields)
        fields = [_tuple_encode(f) for f in fields]
        return namedtuple(type(self).__name__ + 'Tuple', fields)(**new_dict)

    def dump(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError("Path not specified")
        elif not self.loaded:
            logger.info("Skipped dumping unloaded %s", type(self).__name__)

        if path.endswith('.av.bcolz'):
            self._dump_bcolz(path)
        elif path.endswith('.av.npz'):
            self._dump_npz(path)
        else:
            raise ValueError("Invalid path ending: %s" % path)
        self._dump_extra(path)

    def _dump_bcolz(self, path):
        import bcolz as bz
        names, cols = zip(*self.items())
        names = [_tuple_encode(name) for name in names]
        bcols = bz.ctable(cols, names=names, rootdir=path, mode='w')
        bcols.flush()
        bcols.free_cachemem()

    def _dump_npz(self, path):
        np.savez_compressed(path, **self)

    def _dump_extra(self, path):
        self.extra.__len = len(self)
        np.savez_compressed(self._extra_path(path), **self.extra)

    def load(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError("Path not specified")
        elif not os.path.exists(path):
            raise ValueError("Path not found: %s" % path)

        logger.debug("Loading from %s", path)
        if path.endswith('.av.bcolz'):
            self._load_bcolz(path)
        elif path.endswith('.av.npz'):
            self._load_npz(path)
        else:
            raise ValueError("Unknown data type: %s" % path)
        logger.debug("%s columns loaded", len(self._cols))
        self._load_extra(path)
        if not len(self):
            self.set_len(self._guess_len())
        self._tainted(extra=False)

    def _load_bcolz(self, path):
        import bcolz as bz
        if not os.path.isdir(path):
            raise ValueError("Path is not a directory: %s" % path)
        with bz.ctable(rootdir=path) as bcols:
            cols = dict()
            for bcolz_col in bcols.names:
                col = _tuple_decode(bcolz_col)
                logger.debug("Loading column %s", col)
                cols[col] = bcols[bcolz_col][:]
                sleep()
            self.update(cols)

    def _load_npz(self, path):
        if not os.path.isfile(path):
            raise ValueError("Path is not a file: %s" % path)
        with np.load(path) as npz:
            self.update(npz)

    def _load_extra(self, path):
        path_extra = self._extra_path(path)
        if os.path.isfile(path_extra):
            try:
                npz = np.load(path_extra)
            except IOError:
                pass
            else:
                # Should we clear all previous contents of extra before??
                self.extra.update(npz)
            finally:
                npz.close()
            for key, val in self.extra.items():
                if val.ndim == 0:
                    self.extra[key] = val.flat[0]
            org_len = self.extra.pop('__len', None) or self.extra.pop('_len', None)
            if org_len:
                self.set_len(org_len)

    def _guess_len(self):
        return min(len(col) for col in self._cols.values())

    def purge(self):
        if not self.path:
            raise ValueError("No path set")
        if os.path.isdir(self.path) and self.path.endswith('.av.bcolz'):
            shutil.rmtree(self.path)
        elif os.path.isfile(self.path) and self.path.endswith('.av.npz'):
            os.remove(self.path)

        path_extra = self.path_extra
        if path_extra and os.path.isfile(path_extra):
            os.remove(path_extra)

    def _tainted(self, extra=True):
        """ Gets called, if data was modified and caching should be updated"""
        if extra:
            self.extra.clear()

    def freemem(self):
        """ Delete cached data """

    def memsize(self):
        return sum(col.dtype.itemsize for col in self.values()) * self.allocated

    @classmethod
    def prepared(cls, **kwargs):
        """ All defined fields prepared and initialized """
        ac = cls(**kwargs)
        for field in ac.fields:
            ac.create_col(field)
        return ac

    @classmethod
    def empty_like(cls, other):
        copied = cls.prepared(allocation=other.allocated)
        copied.set_len(len(other))
        for col in other:
            copied.create_col(col)
        return copied

    @classmethod
    def from_file(cls, path):
        ac = cls(path=path)
        ac.load()
        return ac

    @classmethod
    def from_tuple(cls, tup):
        ac = cls()
        cols = dict()
        for tup_col in tup._fields:
            cols[_tuple_decode(tup_col)] = getattr(tup, tup_col)[:]
            sleep()
        ac.update(cols)
        return ac
