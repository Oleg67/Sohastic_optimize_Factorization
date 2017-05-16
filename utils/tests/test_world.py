import pytest
import numpy as np

from ..world import (MILE, FURLONG, YARD, STONE, POUND, Float,
                     LimitedFloat, Distance, RaceClass, Going, Price, Weight,
                     Equipment, Handicap, Obstacle, Country)


def test_float():
    assert Float(3.0).value == 3.0
    assert Float(3).value == 3.0
    assert Float(np.float32(3.0)).value == 3.0
    assert Float(np.int64(3.0)).value == 3.0
    assert Float(np.nan).value is np.nan
    pytest.raises(TypeError, Float, 'bla')


def test_limited_float():
    assert LimitedFloat(3.0).value == 3.0
    pytest.raises(ValueError, LimitedFloat, np.nan)


def test_handicap():
    assert Handicap('Cond').value == 'C'
    pytest.raises(ValueError, Handicap, 'bla')


def test_country():
    assert Country('Germany').value == 'DEU'
    assert Country('GBR').value == 'GBR'
    assert Country('IR').value == 'IRL'
    pytest.raises(ValueError, Country, 'Nosuchcountry')


def test_obstacle():
    assert Obstacle('NH Flat').value == 'F'
    assert Obstacle('C').format() == 'Chase'
    pytest.raises(ValueError, Obstacle, 'bla')


def test_equipment():
    assert Equipment('s').name == 'Cheekpieces'
    assert Equipment('B').name == 'Bute'
    pytest.raises(ValueError, Equipment, '-')


def test_price_validation():
    assert Price('5.5').value == 5.5
    assert Price('evens').value == 2
    pytest.raises(ValueError, Price, '0.5')
    pytest.raises(ValueError, Price, 0.9)
    pytest.raises(ValueError, Price, 1020)


def test_price_float_tolerance():
    assert Price(1000.0000000001).value == 1000


vals = [1000.0, 904, 31.1, 25.3, 10.4, 10.1, 7.49, 1.016, 1.011]
rounded = [1000, 900, 32, 25, 10.5, 10, 7.4, 1.02, 1.01]

@pytest.mark.parametrize(("val", "expected"), zip(vals, rounded))
def test_price_rounded(val, expected):
    assert abs(Price(val).rounded - expected) < 1e-10


def test_price_rounding():
    np.testing.assert_array_almost_equal(Price.round(vals), rounded, decimal=10)
    np.testing.assert_array_almost_equal(Price.round([1.113, np.nan]), [1.11, np.nan], decimal=10)


def test_weight():
    assert Weight('8-4').value == 8 * STONE + 4 * POUND
    pytest.raises(ValueError, Weight, '100-5')


@pytest.mark.parametrize(("len_str", "length"),
    ((u'5F 110YDS', 5 * FURLONG + 110 * YARD),
     ('2m4f', 2 * MILE + 4 * FURLONG),
     ('5F', 5 * FURLONG),
     ('4m', 4 * MILE),
     ('3m2f4yds', 3 * MILE + 2 * FURLONG + 4 * YARD),
     ('3m 2f 4y', 3 * MILE + 2 * FURLONG + 4 * YARD)))
def test_distance(len_str, length):
    assert Distance(len_str).meters == length


def test_distance_invalid():
    pytest.raises(ValueError, Distance, 10000)
    pytest.raises(TypeError, Distance, None)


@pytest.mark.parametrize(("test_str", "expected"),
    (('Maiden', 'B'),
     ('Class', 'L')))
def test_race_class(test_str, expected):
    assert RaceClass(test_str).code == expected


def test_race_class_invalid():
    pytest.raises(ValueError, RaceClass, 'bla')
    pytest.raises(ValueError, RaceClass, None)


def test_going_translation():
    assert len(Going.translation) > 10
    assert 7 in Going.translation


@pytest.mark.parametrize(("test_val", "expected"),
    ((7, 'Fast'),
     ('GOOD TO FIRM', 'Good to firm'),
     ('STD-FAST', 'Good to fast'),
     ('STD-FST', 'Good to fast'),
     ('G-F', 'Good to fast')))
def test_going_format(test_val, expected):
    assert Going(test_val).format() == expected


@pytest.mark.parametrize(("test_val", "expected"),
    (('HARD', 'HD'),
     ('GD-FRM', 'GD-FM'),
     (3, 'SF'),
     (7, 'FT'),
     (-4, None),
     (20, None),
     (-2.5, None),
     (343.34, None)))
def test_going_parsing(test_val, expected):
    if expected is None:
        pytest.raises(ValueError, Going, test_val)
    else:
        assert Going(test_val).code == expected
