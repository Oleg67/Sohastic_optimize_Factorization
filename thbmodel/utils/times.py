"""
Some little module, to handle all the stfu python timing hassle.
"""
from __future__ import division

import time
import calendar
from datetime import datetime, date, timedelta
from numbers import Number

from pytz import timezone, UTC


SECOND = 1
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR
WEEK = 7 * DAY
MONTH30 = 30 * DAY
MONTH31 = 31 * DAY
MONTH = MONTH30
YEAR = 365 * DAY


FORMATS = dict(
    de="%d-%m-%Y",
    de_hm="%d-%m-%Y %H:%M",
    de_hms="%d-%m-%Y %H:%M:%S",
    int="%Y-%m-%d",
    int_hm="%Y-%m-%d %H:%M",
    int_hms="%Y-%m-%d %H:%M:%S",
    int_hmsmu="%Y-%m-%d %H:%M:%S.%f",
    uk="%d/%m/%Y",
    uk_short_hm="%d/%m/%y %H:%M",
    uk_short_hms="%d/%m/%y %H:%M:%S",
    uk_hms="%d/%m/%Y %H:%M:%S",
    json="%Y-%m-%dT%H:%M:%S.%fZ",
)

tz_london = timezone('Europe/London')
tz_utc = UTC


def ordinal_day(day):
    if day > 31 or day <= 0:
        raise ValueError('invalid day of month')
    try:
        suffix = ["st", "nd", "rd"][(day - 1) % 10]
    except IndexError:
        suffix = "th"
    return '%d%s' % (day, suffix)


def timeobj_to_float(timeobj, tz=UTC):
    if isinstance(tz, str):
        tz = timezone(tz)
    if not timeobj.tzinfo:
        timeobj = tz.localize(timeobj)
    return float(calendar.timegm(timeobj.utctimetuple())) + timeobj.microsecond / 1e6


def dateobj_to_float(timeobj):
    return float(calendar.timegm(timeobj.timetuple()))


def timestr_to_float(timestr, fmt=None, tz=UTC):
    if isinstance(tz, str):
        tz = timezone(tz)
    fmt = FORMATS.get(fmt, fmt)
    if not fmt:
        # Try to guess format
        length = len(timestr)
        if length == 10:
            if timestr[2] == '/':
                fmt = FORMATS['uk']
            elif timestr[4] == '-':
                fmt = FORMATS['int']
            else:
                fmt = FORMATS['de']
        elif length == 14:
            fmt = FORMATS['uk_short_hm']
        elif length == 16:
            if timestr[2] == '-':
                fmt = FORMATS['de_hm']
            else: fmt = FORMATS['int_hm']
        elif length == 17:
            fmt = FORMATS['uk_short_hms']
        elif length == 19:
            if timestr[2] == '/':
                fmt = FORMATS['uk_hms']
            elif timestr[2] == '-':
                fmt = FORMATS['de_hms']
            else: fmt = FORMATS['int_hms']
        elif length > 20 and 'T' in timestr and  timestr.endswith('Z'):
            fmt = FORMATS['json']
        elif length > 20:
            fmt = FORMATS['int_hmsmu']
        else:
            raise ValueError("Time format not recognized: %s" % timestr)
    timeobj = tz.localize(datetime.strptime(timestr, fmt))
    if timeobj.year > 2030 and '%y' in fmt:
        # Sanitize unclear century data
        timeobj = timeobj.replace(year=timeobj.year - 100)
    return timeobj_to_float(timeobj)


def datenum_to_float(datenum):
    """ Convert a matlab datenum to a unix timestamp """
    # See here for reference: http://sociograph.blogspot.co.uk/2011/04/how-to-avoid-gotcha-when-converting.html
    timeobj = datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)
    return timeobj_to_float(timeobj)


class timestamp(Number):
    __slots__ = '_timestamp',

    timeitems = dict(microsecond=0, second=1, minute=2, hour=3, day=4, month=5, year=6)

    def __init__(self, t=None, fmt=None, tz=UTC):
        if t is None:
            self._timestamp = time.time()
        elif isinstance(t, float):
            self._timestamp = t
        elif isinstance(t, basestring):
            self._timestamp = timestr_to_float(t, fmt=fmt, tz=tz)
        elif isinstance(t, datetime):
            self._timestamp = timeobj_to_float(t, tz=tz)
        elif isinstance(t, date):
            self._timestamp = dateobj_to_float(t)
        else:
            try:
                self._timestamp = float(t)
            except TypeError:
                raise TypeError("Cannot convert type %s to timestamp" % type(t).__name__)

    @classmethod
    def from_matlab(cls, datenum):
        return cls(datenum_to_float(datenum))

    def format(self, fmt='int_hm', tz=None):  # @ReservedAssignment
        """ Ordentlicher Ersatz fuer time.asctime(time.gmtime(stamp)) """
        timeobj = self.localized(tz) if tz else self.timeobj()
        return timeobj.strftime(FORMATS.get(fmt, fmt))

    def __repr__(self):
        return "<Timestamp %s>" % self.format('int_hmsmu')

    def __str__(self):
        return self.format('int_hmsmu')

    def startof(self, timeitem):
        if timeitem not in self.timeitems:
            raise ValueError("Time period '%s' not in  %s" %
                             (timeitem, ','.join(sorted(self.timeitems, key=lambda x: self.timeitems[x], reverse=True))))
        maxpos = self.timeitems[timeitem]
        replace_dict = dict((item, 1 if item in ('day', 'month') else 0)
                            for item, pos in self.timeitems.items() if pos < maxpos)
        replace_dict['tzinfo'] = UTC
        dt = datetime.utcfromtimestamp(self._timestamp).replace(**replace_dict)
        return type(self)(timeobj_to_float(dt))

    @classmethod
    def validated(cls, x):
        """ Make timestamp behave like one of the SingleValue world objects. """
        return cls(x)._timestamp

    def daytime(self):
        return self._timestamp % DAY

    def daystart(self):
        return type(self)(self._timestamp - self.daytime())

    def monthstart(self):
        return self.startof('month')

    def yearstart(self):
        return self.startof('year')

    def timeobj(self):
        return datetime.utcfromtimestamp(self._timestamp).replace(tzinfo=UTC)

    def decimaldate(self):
        timeobj = self.timeobj()
        return timeobj.year * 10000 + timeobj.month * 100 + timeobj.day

    def __getattr__(self, attr):
        """ Return values like year, month and day """
        if attr in self.timeitems:
            return getattr(self.timeobj(), attr)
        else:
            raise AttributeError(attr)

    def timetuple(self):
        return self.timeobj().utctimetuple()

    def dayofyear(self):
        return self.timetuple().tm_yday

    def weekday(self):
        return self.timetuple().tm_wday

    def localized(self, tz):
        if isinstance(tz, basestring):
            tz = timezone(tz)
        return self.timeobj().astimezone(tz)

    #### Let's behave like an ordinary float ####

    def __float__(self):
        return self._timestamp

    def __int__(self):
        return int(self._timestamp)

    @staticmethod
    def _convert(other):
        if isinstance(other, timedelta):
            return other.total_seconds()
        elif isinstance(other, datetime):
            return timeobj_to_float(other)
        elif isinstance(other, date):
            return dateobj_to_float(other)
        return other

    def __add__(self, other):
        return type(self)(self._timestamp + self._convert(other))

    def __sub__(self, other):
        return type(self)(self._timestamp - self._convert(other))

    def __mul__(self, other):
        return type(self)(self._timestamp * other)

    def __neg__(self):
        return type(self)(-self._timestamp)

    def __truediv__(self, other):
        return type(self)(self._timestamp / other)

    def __floordiv__(self, other):
        return type(self)(self._timestamp // other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return type(self)(self._convert(other) - self._timestamp)

    def __rtruediv__(self, other):
        return type(self)(other / self._timestamp)

    def __rfloordiv__(self, other):
        return type(self)(other // self._timestamp)

    def __iadd__(self, other):
        self._timestamp += self._convert(other)
        return self

    def __isub__(self, other):
        self._timestamp -= self._convert(other)
        return self

    def __imul__(self, other):
        self._timestamp *= other
        return self

    def __itruediv__(self, other):
        self._timestamp /= other
        return self

    def __ifloordiv__(self, other):
        self._timestamp //= other
        return self

    def __eq__(self, other):
        return self._timestamp == self._convert(other)

    def __ne__(self, other):
        return self._timestamp != self._convert(other)

    def __gt__(self, other):
        return self._timestamp > self._convert(other)

    def __lt__(self, other):
        return self._timestamp < self._convert(other)

    def __ge__(self, other):
        return self._timestamp >= self._convert(other)

    def __le__(self, other):
        return self._timestamp <= self._convert(other)


class Timer(object):
    __slots__ = 'started', 'stopped', 'verbose'

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.started = self.stopped = None

    @property
    def elapsed(self):
        try:
            return (self.stopped or time.time()) - self.started
        except TypeError:
            raise ValueError("Timer must be started first")

    def start(self):
        self.stopped = None
        self.started = time.time()

    def stop(self, *args):
        self.stopped = time.time()
        if self.verbose:
            print 'elapsed time: %s' % timedelta(seconds=self.elapsed)

    def __enter__(self):
        self.start()
        return self

    __exit__ = stop

