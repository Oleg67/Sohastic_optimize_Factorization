import pytest
import time
from datetime import datetime, date
import calendar

from ..times import timestamp, timezone, ordinal_day, DAY, HOUR

def ref_time(*args):
    timeobj = datetime(*args)
    return calendar.timegm(timeobj.timetuple())


def test_init_from_float():
    t = time.time()
    assert float(timestamp(t)) == t


def test_init_from_date():
    assert float(timestamp(date(*datetime.utcnow().timetuple()[:3]))) == timestamp().daystart()


def test_init_from_datetime():
    d = datetime(2006, 2, 1, 0, 0)
    t = timestamp(d)
    assert t == ref_time(2006, 2, 1)


@pytest.mark.parametrize("ts",
    ["2006-02-01",
     "2006-02-01 00:00",
     "2006-02-01 00:00:00",
     "2006-02-01 00:00:00.000000",
     "01/02/2006",
     "01/02/2006 00:00:00",
     "2006-02-01T00:00:00.000Z",
     "2006-02-01T00:00:00.000000Z"])
def test_init_from_string(ts):
    stamp = timestamp(ts)
    assert stamp == ref_time(2006, 2, 1)
    assert str(stamp) == "2006-02-01 00:00:00.000000"


@pytest.mark.parametrize(["ts", "fmt"],
    [("27/01/201604:40", "%d/%m/%Y%H:%M"),
     ("201601270440", "%Y%m%d%H%M"),
     ("27/01/16 04:40", "%d/%m/%y %H:%M")])
def test_init_custom_format(ts, fmt):
    assert timestamp(ts, fmt=fmt) == ref_time(2016, 1, 27, 4, 40)


def test_copying():
    now = timestamp()
    assert timestamp(now) == now
    assert timestamp(now) is not now


def test_format_with_timezone():
    stamp = timestamp('2015-07-15 16:30')
    assert stamp.format("%H") == '16'
    assert stamp.format("%H", tz=timezone('Europe/London')) == '17'


def test_daytime():
    assert timestamp('1990-11-11 10:00').daytime() == 10 * HOUR
    assert timestamp('1950-01-01 10:00').daytime() == 10 * HOUR


def test_epoch():
    assert timestamp('1970-01-01 00:00') == 0


def test_timestamp_before_epoch():
    assert timestamp('1965-01-01') == -157766400


def test_timediff():
    assert timestamp('1970-01-01 10:00') - timestamp('1970-01-01 00:00') == 10 * 60 * 60


def test_invalid_hour():
    pytest.raises(ValueError, timestamp, '1970-01-01 25:00')


def test_invalid_day():
    pytest.raises(ValueError, timestamp, '1970-01-50 00:00')


def test_invalid_month():
    pytest.raises(ValueError, timestamp, '1970-13-01 00:00')


def test_matlab():
    datenum = 731965.04835648148
    assert timestamp.from_matlab(datenum).daystart() == timestamp.from_matlab(int(datenum))
    assert timestamp.from_matlab(datenum).timetuple()[:3] == (2004, 1, 19)
    assert timestamp.from_matlab(int(datenum) + 0.5).timetuple()[3:6] == (12, 0, 0)


@pytest.mark.parametrize("d", range(32))
def test_ordinal_day(d):
    if d == 0:
        pytest.raises(ValueError, ordinal_day, 0)
    elif d % 10 == 1:
        assert ordinal_day(d).endswith("st")
    elif d % 10 == 2:
        assert ordinal_day(d).endswith("nd")
    elif d % 10 == 3:
        assert ordinal_day(d).endswith("rd")
    else:
        assert ordinal_day(d).endswith("th")


def test_timezone_diff():
    tz = timezone('Europe/London')
    dt1 = tz.localize(datetime(2012, 3, 24, 0, 0))
    dt2 = tz.localize(datetime(2012, 3, 26, 0, 0))
    diff = float(timestamp(dt2) - timestamp(dt1))
    assert diff == 47 * 60 * 60


@pytest.mark.parametrize("tz_str", ["Europe/London", "Europe/Berlin", "Europe/Kiev"])
def test_localized(tz_str):
    tz = timezone(tz_str)

    now = time.time()
    for test_time in (now, now + 180 * DAY):
        ref_offset = tz.utcoffset(datetime.utcfromtimestamp(test_time)).seconds / 3600
        assert 0 <= ref_offset <= 3
        assert timestamp(test_time).localized(tz).hour == (timestamp(test_time).timeobj().hour + ref_offset) % 24


@pytest.mark.parametrize(["timedata", "ref"],
    (("2012-01-26 17:05", "2012-01-26 17:05"),  # winter time
     ("2012-07-04 17:50", "2012-07-04 16:50"),  # summer time
     (datetime(2012, 1, 26, 17, 5), "2012-01-26 17:05"),  # winter time
     (datetime(2012, 7, 4, 17, 50), "2012-07-04 16:50")))  # summer time
def test_summer_time(timedata, ref):
    tz = 'Europe/London'
    ts = timestamp(timedata, tz=tz)
    assert str(timestamp(ts)).startswith(ref)


def test_add():
    assert timestamp("2016-01-01 10:00") + HOUR == timestamp("2016-01-01 11:00")


def test_iadd():
    ts = timestamp("2016-01-01 10:00")
    ts += HOUR
    assert ts == timestamp("2016-01-01 11:00")


def test_sub():
    assert timestamp("2016-01-01 10:00") - HOUR == timestamp("2016-01-01 09:00")


def test_isub():
    ts = timestamp("2016-01-01 10:00")
    ts -= HOUR
    assert ts == timestamp("2016-01-01 09:00")


def test_eq():
    assert timestamp("2016-01-01 10:00") == timestamp("2016-01-01 10:00")
    assert not timestamp("2016-01-01 10:00") == timestamp("2016-01-01 11:00")


def test_ne():
    assert not timestamp("2016-01-01 10:00") != timestamp("2016-01-01 10:00")
    assert timestamp("2016-01-01 10:00") != timestamp("2016-01-01 11:00")


def test_gt():
    assert not timestamp("2016-01-01 10:00") > timestamp("2016-01-01 11:00")
    assert not timestamp("2016-01-01 10:00") > timestamp("2016-01-01 10:00")
    assert timestamp("2016-01-01 11:00") > timestamp("2016-01-01 10:00")


def test_ge():
    assert not timestamp("2016-01-01 10:00") >= timestamp("2016-01-01 11:00")
    assert timestamp("2016-01-01 10:00") >= timestamp("2016-01-01 10:00")
    assert timestamp("2016-01-01 11:00") >= timestamp("2016-01-01 10:00")


def test_lt():
    assert timestamp("2016-01-01 10:00") < timestamp("2016-01-01 11:00")
    assert not timestamp("2016-01-01 10:00") < timestamp("2016-01-01 10:00")
    assert not timestamp("2016-01-01 11:00") < timestamp("2016-01-01 10:00")


def test_le():
    assert timestamp("2016-01-01 10:00") <= timestamp("2016-01-01 11:00")
    assert timestamp("2016-01-01 10:00") <= timestamp("2016-01-01 10:00")
    assert not timestamp("2016-01-01 11:00") <= timestamp("2016-01-01 10:00")
