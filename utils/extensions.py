import os
import time
import csv
import itertools
import numpy as np
import pickle

import texttable


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
