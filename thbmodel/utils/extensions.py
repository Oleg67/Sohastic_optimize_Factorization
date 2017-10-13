import os
import time
import csv
import itertools
import numpy as np
import pickle

import texttable


class FilteredPickler(pickle.Pickler):
    """ Pickler, which does not cache the IDs of certain object types.
        Greatly reduces the memory footprint, when storing huge containers
        with unique contents.
    """

    def __init__(self, file, protocol=None, cached_classes=object, fast=False):
        pickle.Pickler.__init__(self, file, protocol)
        self.cached_classes = cached_classes
        self.fast = fast

    def memoize(self, obj):
        if self.fast or not isinstance(obj, self.cached_classes):
            return
        assert id(obj) not in self.memo
        memo_len = len(self.memo)
        self.write(self.put(memo_len))
        self.memo[id(obj)] = memo_len, obj


class ConstantComparisonResult(object):
    """ Infinite range delimiter for BTree searches """

    def __init__(self, result, name=None):
        self.result = result
        self.name = name

    def __cmp__(self, x):
        return self.result

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return '<%s>' % self.name

Larger = ConstantComparisonResult(1, 'Larger')
Equal = ConstantComparisonResult(0, 'Equal')
Smaller = ConstantComparisonResult(-1, 'Smaller')


class ConstantContainingObject(object):
    def __init__(self, result, name=None):
        self.result = result
        self.name = name

    def __contains__(self, x):
        return self.result

    def __len__(self):
        return np.inf

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return '<%s>' % self.name

Everything = ConstantContainingObject(True, 'Everything')
Nothing = ConstantContainingObject(False, 'Nothing')


class DummyLock(object):
    """ Some lock that does not lock anything """
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

    def acquire(self):
        return

    def release(self):
        return


class CleanTextTable(texttable.Texttable):
    """ Texttable mit weniger verschnoerkelten defaults und floats werden automatisch gekuerzt """

    def __init__(self, max_width=120, default="%.4g"):
        texttable.Texttable.__init__(self, max_width=max_width)
        self.set_deco(self.VLINES | self.HEADER)
        self.set_chars(tuple('-|--'))
        self.default = default

    def draw(self):
        return texttable.Texttable.draw(self) + '\n'

    @staticmethod
    def format_field(item, fieldname=None, default="%.4g"):
        fieldname = '' if fieldname is None else fieldname.upper()
        if isinstance(item, float):
            if np.isnan(item):
                return 'NaN'
            if fieldname in ('BPR', 'LPR') and item < 1000:
                fmt_str = "%.3g"
            elif fieldname in ('VOLUME',):
                fmt_str = "%.0f"
            elif fieldname in ('PL_M', 'PL_E', 'PL_R', 'WAG'):
                fmt_str = "%.2f"
            elif fieldname in ('LTIMESTAMP', 'TIMESTAMP', 'TIME'):
                fmt_str = "%.1f"
            elif (fieldname.endswith('_E') or fieldname in ('UKHR_V',)) and item < 1000:
                fmt_str = "%.3g"
            elif fieldname in ('S1', 'MPROB', 'S2', 'ER', 'WTB'):
                fmt_str = "%.4f"
            elif fieldname in ('FRAC',):
                fmt_str = "%.5f"
            elif fieldname in ('PROB',):
                fmt_str = "%.7f"
            else:
                fmt_str = default
            item_str = fmt_str % item
            if '.' in item_str and fieldname not in ('LTIMESTAMP', 'TIMESTAMP', 'TIME'):
                item_str = item_str.rstrip('0').rstrip('.')
            return item_str
        elif isinstance(item, str):
            return item
        else:
            return str(item)

    def add_row(self, row):
        new_row = [self.format_field(item, fieldname, self.default)
                   for item, fieldname in itertools.izip(row, self._header)]
        self._check_row_size(new_row)
        self._rows.append(map(str, new_row))


class cached(object):
    """ Cache the result of a property calculation forever """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__module__ = func.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        cached_attr = "_v_%s_cached" % self.__name__
        try:
            return getattr(obj, cached_attr)
        except AttributeError:
            value = self.func(obj)
            setattr(obj, cached_attr, value)
            return value


class cached_ttl(object):
    """ As seen at http://wiki.python.org/moin/PythonDecoratorLibrary. Sample use:

        class SomeClass(object):
            @cached_ttl(30)
            def someprop(self):
                print 'Actually calculating value'
                return 13
    """
    def __init__(self, ttl=0):
        assert ttl >= 0
        self.ttl = ttl

    def __call__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__module__ = func.__module__
        return self

    def __get__(self, obj, owner):
        if obj is None:
            return self

        cached_attr = "_v_%s_cached" % self.__name__
        last_update_attr = "_v_%s_last_update" % self.__name__
        now = time.time()
        try:
            value = getattr(obj, cached_attr)
            if self.ttl > 0 and now - getattr(obj, last_update_attr, 0) > self.ttl:
                raise AttributeError
        except AttributeError:
            value = self.func(obj)
            setattr(obj, cached_attr, value)
            setattr(obj, last_update_attr, now)
        return value




class CSVWriter(object):
    """ Extension for csv.writer, allowing file resume, 
        different string conversions and initialization.
    """
    def __init__(self, fpath, fieldnames, resume=False):
        self.fieldnames = fieldnames
        if resume and os.path.exists(fpath):
            f = open(fpath, 'a')
            fsize = os.fstat(f.fileno()).st_size
            self.writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            if fsize == 0:
                self.writer.writerow(self.fieldnames)
        else:
            f = open(fpath, 'w')
            self.writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(self.fieldnames)
        self.csv_file = f

    @staticmethod
    def _str_float(fl):
        return ("%.4f" % fl).rstrip('0').rstrip('.')

    def writerow(self, row):
        row = [self._str_float(item) if isinstance(item, float) else item for item in row]
        ret = self.writer.writerow(row)
        self.csv_file.flush()
        return ret

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.csv_file.close()
