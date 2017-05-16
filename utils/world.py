"""
A bunch of representations for betting related real live objects.
"""
from __future__ import division
import re
from collections import namedtuple

import numpy as np
import pycountry

from .errors import ValidationValueError


# Retarded length units for retarded retards
YARD = 0.9144  # Exact definition
FURLONG = 220 * YARD
MILE = 8 * FURLONG
FOOT = YARD / 3
INCH = YARD / 36

POUND = 0.45359237  # in kg
STONE = 14 * POUND

# Even more retarded length units
HORSE_LENGTH = 8 * FOOT
NECK = HORSE_LENGTH / 3
SHORT_NECK = NECK * 0.75  # Roughly estimated
HEAD = NECK / 2
SHORT_HEAD = HEAD / 2
NOSE = HEAD / 3


class SingleValue(object):
    """ Things that can as well be expressed as a single value """

    __slots__ = '_value',


    def __init__(self, value):
        self._value = value

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, self._value)

    @property
    def value(self):
        return self._value

    @classmethod
    def validated(cls, val, **kwargs):
        try:
            return cls(val, **kwargs).value
        except ValueError as e:
            raise ValidationValueError(str(e))


class Codified(SingleValue):
    """ Things that can as well be expressed as string codes """
    def __str__(self):
        return self._value

    @property
    def code(self):
        return self._value


class Country(Codified):
    exceptions = dict((
        # Racingsense
         ('UNR', 'UNR'),
         ('ENG', 'GBR'),
         ('SPA', 'ESP'),
         ('A.ARAB', 'ARE'),
         ('A.A', 'ARE'),
         ('BAR', 'BRA'),
         ('SU', 'SDN'),
         ('HOL', 'NLD'),
         ('SAF', 'ZAF'),
         ('DEN', 'DNK'),
         ('SWI', 'CHE'),
         ('ZIM', 'ZWE'),
         ('YUG', 'SRB'),
         ('MOR', 'MAR'),
         ('GDR', 'DEU'),
         ('CHR', 'CZE'),
         ('SLO', 'SVK'),

        # Betfair
         ('BRZ', 'BRA'),
         ('CHI', 'CHL'),
         ('IR', 'IRL'),
         ('IRE', 'IRL'),
         ('GER', 'DEU'),
         ('URU', 'URY'),
         ('UK', 'GBR'),
         ('RSA', 'ZAF'),
         ('UAE', 'ARE'),
         ('JAP', 'JPN'),
         ('SPN', 'ESP'),
         ('ITY', 'ITA'),
         ('SUI', 'CHE'),
         ('SIN', 'SGP'),

         # Attheraces
         ('SNL', 'SWZ'),
    ))

    def __init__(self, code):
        if not isinstance(code, basestring):
            raise TypeError("Invalid country type: %s" % type(code).__name__)
        code = code.strip().upper()
        if code in self.exceptions:
            self._value = self.exceptions[code]
            return

        if len(code) == 2:
            searchkey = 'alpha2'
        elif len(code) == 3:
            searchkey = 'alpha3'
        else:
            searchkey = 'name'
            code = code.capitalize()
        try:
            self._value = pycountry.countries.get(**{searchkey: code}).alpha3
        except KeyError:
            raise ValueError("Invalid country code: %s" % code)

    @property
    def name(self):
        return pycountry.countries.get(alpha3=self._value).name

    country = name


class Obstacle(Codified):
    # TODO: This needs to be accompanied by a race type or be more detailed in that way
    FLAT = 'F'
    HURDLE = 'H'
    CHASE = 'C'
    STEEPLE_CHASE = 'S'  # American nomenclature, same as Chase/NH

    difficulty = dict((
        # Racing Sense
        ('AW', 'F'),  # Not nice, but should mainly get correct results
        ('FL', 'F'),
        ('NHBU', 'F'),
        ('NHHU', 'H'),
        ('NHCH', 'C'),

        # Betfair
        ('NHF', 'F'),
        ('INHF', 'F'),
        ('Flat', 'F'),
        ('Hrd', 'H'),
        ('Chs', 'C'),
        ('Stpl', 'C'),

        # Timeform / Ukhr
        ('Bumper', 'F'),
        ('NH Flat', 'F'),
        ('Chase', 'C'),
        ('Hurdle', 'H'),
        ('Steeple Chase', 'C'),
        ('Hunter Chase', 'C'),

        ('HURDLE', 'H'),
        ('CHASE', 'C'),
        ('NATIONAL HUNT FLAT', 'F'),
        ('NH FLAT', 'F'),
        ('FLAT', 'F'),
        ('STEEPLE CHASE', 'C'),
        ('HUNTER CHASE', 'C'),
    ))

    readable = dict((
        (FLAT, 'Flat'),
        (HURDLE, 'Hurdle'),
        (CHASE, 'Chase'),
        (STEEPLE_CHASE, 'Steeple Chase'),
    ))

    handles = frozenset((FLAT, HURDLE, CHASE))

    def __init__(self, obstacle):
        if len(obstacle) == 1:
            obstacle = obstacle.upper()
        if obstacle in self.handles:
            self._value = obstacle
        else:
            try:
                self._value = self.difficulty[obstacle]
            except KeyError:
                raise ValueError('Invalid obstacle: %s' % obstacle)

    def format(self):
        return self.readable.get(self._value)


class Handicap(Codified):
    NONE = 'N'
    HANDICAP_USUAL = 'H'
    WEIGHT_FOR_AGE = 'A'
    CONDITIONS = 'C'
    WELTER = 'W'

    valid = frozenset((NONE, HANDICAP_USUAL, WEIGHT_FOR_AGE, WELTER, CONDITIONS))

    names = dict((
        ('None', 'N'),
        ('-', 'N'),
        ('', 'N'),
        ('Hcap', 'H'),  # Weight applied by an official handicapper
        ('WFA', 'A'),  # Weight for age
        ('Wltr', 'W'),
        ('Cond', 'C'),
        ('Conditions', 'C'),
        ('Stks', 'C'),  # Betfair and US usage for conditions
        ('Stakes', 'C'),
    ))

    def __init__(self, handicap):
        if handicap in self.valid:
            self._value = handicap
        else:
            try:
                self._value = self.names[handicap]
            except KeyError:
                raise ValueError('Wrong handicap value: %s' % handicap)


class RaceClass(Codified):
    """ Race class classifies races by their intention and the general skill level of the competitors """

    BUMPER = 'B'  # Implies flat, probably amateur jockeys, low stakes, bad horses, unexperienced young horses
    SELLING = 'S'
    CLAIMING = 'M'
    LISTED = 'L'
    GRADE3 = 'G3'
    GRADE2 = 'G2'
    GRADE1 = 'G1'  # Gaul-Bundesliga, implies WFA handicapping
    # STAKES = 'S' -> only includes conditions handicapping
    # GROUPx: GRADEx, difference just in obstacle type
    # CLASSIFIED: Graded + Listed
    # CLASSIFIED implies weighting type CONDITIONS

    names = dict((
        ('Maiden', BUMPER),
        ('Novice', BUMPER),
        ('Beginner', BUMPER),

        ('Claiming', CLAIMING),
        ('Claim', CLAIMING),

        ('Selling', SELLING),
        ('Sell', SELLING),
        ('Auction', SELLING),
        ('Classified', LISTED),
        ('Class', LISTED),
        ('Grd3', GRADE3),
        ('Grp3', GRADE3),
        ('Grd2', GRADE2),
        ('Grp2', GRADE2),
        ('Grd1', GRADE1),
        ('Grp1', GRADE1),
        ('Grand National', GRADE1),
    ))

    handles = frozenset(names.values())

    def __init__(self, raceclass):
        if raceclass in self.handles:
            self._value = raceclass
        else:
            try:
                self._value = self.names[raceclass]
            except KeyError:
                raise ValueError('Wrong race class value: %s' % raceclass)


class Equipment(Codified):
    aliases = dict(s='cp', V='cp', Z='t', C='B', M='L')
    abbreviations = dict(
        b='Blinkers',
        es='Eyeshield',
        h='Hood',
        cp='Cheekpieces',  # Sheepskin
        t='Tongue Tie',  # Tongue Strap
        v='Visor',

        # Used by equibase
        N='No Whip',
        A='Aluminum Pads',
        R='Bar Shoe',
        S='Nasal Strip',
        C='Mud Calks',
        F='Front Bandages',
        Y='No Shoes',
        G='Goggles',
        K='Flipping Halter',

        # Drugs, used by equibase
        B='Bute',
        L='Lasix')

    def __init__(self, equipment):
        unaliased = self.aliases.get(equipment, equipment)
        if unaliased in self.abbreviations:
            self._value = unaliased
        else:
            raise ValueError('Unrecognized equipment: %s' % unaliased)

    @property
    def name(self):
        return self.abbreviations[self._value]


class Ground(object):
    TURF = 'TF'
    DIRT = 'DT'
    POLYTRACK = 'PT'
    EQUITRACK = FIBRESAND = 'ET'
    TAPETA = 'TA'
    ALL_WEATHER = SYNTHETIC = 'AW'
    ALL = 'ALL'

    grounds = TURF, DIRT, SYNTHETIC, ALL_WEATHER

    translation = dict((g, g) for g in grounds)
    translation.update(dict((
        ('AWT', ALL_WEATHER),
        ('ALL WEATHER', ALL_WEATHER),
        ('TURF', TURF),
        ('DIRT', DIRT),
        ('POLYTRACK', POLYTRACK),
        ('EQUITRACK', EQUITRACK),
        ('SYNTHETIC', ALL_WEATHER),
    )))


GoingType = namedtuple('GoingType', 'abbreviation name ground factor description')


class Going(Codified):
    ALL_WEATHER = GoingType('AW', 'All weather', Ground.ALL_WEATHER, 0.05, 'Constant surface for all weather conditions')
    HARD = GoingType('HD', 'Hard', Ground.TURF, -0.28, 'Surface is hard and horses do not have normal cushion of the course')
    FIRM = GoingType('FM', 'Firm', Ground.TURF, -0.58, 'Completely dry turf, will usually produces the fastest times')
    GOOD = GoingType('GD', 'Good', Ground.ALL, 0.05, 'Relatively firm course that contains a slight bit of moisture')
    SOFT = GoingType('SF', 'Soft', Ground.TURF, 0.60, 'Course that contains more moisture than a good track')
    YIELDING = GoingType('YD', 'Yielding', Ground.TURF, 1.20, 'Course with a considerable amount of water and very yielding to the riders running, producing some particularly slow race times')
    HEAVY = GoingType('HY', 'Heavy', Ground.ALL, 1.60, 'A waterlogged heavy course that produces the slowest course times')

    FAST = GoingType('FT', 'Fast', Ground.DIRT, -0.58, 'Horses will usually run their fastest times under these conditions, the surface will run over dry and steady to hard')
    WET_FAST = GoingType('WF', 'Wet-Fast', Ground.DIRT, -0.28, 'This surface will usually produce fast times but will have a slight layer of moisture over the top layer of dirt')
    SLOW = GoingType('SL', 'Slow', Ground.DIRT, 0.60, 'This surface is usually rather deep, and drying out which produces slower than a "Good" track time table')
    SLOPPY = GoingType('SY', 'Sloppy', Ground.DIRT, 0.60, 'As the track continues to accumulate moisture, the base is still solid but water is beginning to seep into the base. Surface water is evident')
    MUDDY = GoingType('MY', 'Muddy', Ground.DIRT, 1.20, 'Moisture has permeated the base of the track, times are somewhat slower and running tires the horses more. ')
    FROZEN = GoingType('FZ', 'Frozen', Ground.DIRT, 0.30, 'As a result of sustained low temperatures, ice particles have formed on the racing surface')

    GOOD_FAST = GoingType('GD-FT', 'Good to fast', Ground.DIRT, -0.30, '')
    GOOD_SLOW = GoingType('GD-SL', 'Good to slow', Ground.DIRT, 0.30, '')
    GOOD_FIRM = GoingType('GD-FM', 'Good to firm', Ground.TURF, -0.30, '')
    GOOD_SOFT = GoingType('GD-SF', 'Good to soft', Ground.TURF, 0.30, '')
    GOOD_YIELD = GoingType('GD-YD', 'Good to yielding', Ground.TURF, 0.50, '')
    YIELD_SOFT = GoingType('YD-SF', 'Yielding to soft', Ground.TURF, 0.90, '')
    SOFT_HEAVY = GoingType('SF-HY', 'Soft to heavy', Ground.TURF, 1.20, '')

    goings = (ALL_WEATHER, HARD, FIRM, GOOD, SOFT, YIELDING, HEAVY, FAST, WET_FAST, SLOW, SLOPPY, MUDDY, FROZEN,
            GOOD_FAST, GOOD_SLOW, GOOD_FIRM, GOOD_SOFT, GOOD_YIELD, YIELD_SOFT, SOFT_HEAVY)

    translation = dict((g.abbreviation, g) for g in goings)
    translation.update((g.name.upper(), g) for g in goings)
    translation.update((
        ('SYNTHETIC', ALL_WEATHER),
        ('AWT', ALL_WEATHER),
        ('STD', GOOD),
        ('STAND', GOOD),
        ('STANDARD', GOOD),
        ('GD-SFT', GOOD_SOFT),
        ('STANDARD TO SOFT', GOOD_SOFT),
        ('STANDARD TO SLOW', GOOD_SLOW),
        ('STANDARD TO FAST', GOOD_FAST),
        ('STANDARD TO FIRM', GOOD_FIRM),
        ('STANDARD TO YIELDING', GOOD_YIELD),
        ('STD-SLOW', GOOD_SLOW),
        ('STD-SLW', GOOD_SLOW),
        ('STD-FAST', GOOD_FAST),
        ('STD-FST', GOOD_FAST),
        ('SFT-HVY', SOFT_HEAVY),
        ('YIELD', YIELDING),
        ('HRD', HARD),
        ('GD-FRM', GOOD_FIRM),
        ('FRM', FIRM),
        ('FST', FAST),
        ('G-F', GOOD_FAST),
        ('G', GOOD),
        ('G-Y', GOOD_YIELD),
        ('GD-YLD', GOOD_YIELD),
        ('Y', YIELDING),
        ('Y-S', YIELD_SOFT),
        ('YLD-SFT', YIELD_SOFT),
        ('S', SLOW),
        ('HVY', HEAVY),
        (7, FAST),
        (6, HARD),
        (5, GOOD_FAST),
        (4, GOOD),
        (3, SOFT),
        (2, YIELDING),
        (1, HEAVY),
    ))

    def __init__(self, going):
        if isinstance(going, (str, unicode)):
            try:
                self._value = self.translation[going.strip().upper()].abbreviation
            except KeyError:
                raise ValueError('Invalid going string: %s' % going)
        elif isinstance(going, int):
            try:
                self._value = self.translation[going].abbreviation
            except KeyError:
                raise ValueError('Invalid going constant: %s' % going)
        else:
            raise ValueError('Invalid going specification: %s' % repr(going))

    def format(self):
        return self.translation[self._value].name


class Float(SingleValue):
    def __init__(self, value):
        self._set_value(value)

    def _set_value(self, value):
        if not isinstance(value, (int, float, np.floating, np.integer)):
            raise TypeError("Invalid input type: %s" % type(value).__name__)
        self._value = float(value)

    def __float__(self):
        return self._value

    def __int__(self):
        return int(self._value)


class LimitedFloat(Float):
    min = float('-inf')
    max = float('+inf')
    precision = None

    def _set_value(self, value):
        if not isinstance(value, (int, float, np.floating, np.integer)):
            raise TypeError("Invalid input type: %s" % type(value).__name__)

        if self.precision:
            value = round(value, self.precision)
        if self.min <= value <= self.max:
            self._value = float(value)
        else:
            raise ValueError("%s not within limits: %s" % (type(self).__name__, value))


class Distance(LimitedFloat):
    matcher = re.compile(r"((?P<miles>\d+)m)?\s*((?P<furlongs>\d+)f)?\s*((?P<yards>\d+)y(ds)?)?$", re.IGNORECASE)
    min = 100  # Recently seen in US racing
    max = 8000
    precision = 6

    def __init__(self, distance):
        if isinstance(distance, (float, int)):
            self._set_value(distance)
        else:
            matched = self.matcher.match(distance)
            if not matched:
                raise ValueError("Could not read distance: %s" % distance)
            matched = matched.groupdict()
            miles = int(matched.get('miles') or 0)
            if miles > 10:
                # Dist was given in meters, not miles
                self._set_value(miles)
            else:
                furlongs = int(matched.get('furlongs') or 0)
                yards = int(matched.get('yards') or 0)
                self._set_value(miles * MILE + furlongs * FURLONG + yards * YARD)

    @property
    def meters(self):
        return self._value


class Weight(LimitedFloat):
    """ Weight in KG """
    min = 35.0
    max = 200

    def __init__(self, weight, delimiter=r'-'):
        if isinstance(weight, basestring):
            splt = weight.strip().split(delimiter)
            try:
                weight = int(splt[0]) * STONE + int(splt[1]) * POUND
            except (ValueError, IndexError):
                raise ValueError("Invalid weight format: %s" % weight)
        self._set_value(weight)


class Price(LimitedFloat):
    """ Accepts decimal and fractional odds that look like following: 5/1, 11/4, 3/1 etc """
    steps = np.concatenate((np.arange(1.01, 2, 0.01), np.arange(2.0, 3.0, 0.02),
                            np.arange(3.0, 4.0, 0.05), np.arange(4, 6, 0.1), np.arange(6, 10, 0.2),
                            np.arange(10, 20, 0.5), np.arange(20, 30, 1), np.arange(30, 50, 2),
                            np.arange(50, 100, 5), np.arange(100, 1001, 10)))
    special_names = {'evens': 2, 'even money': 2, 'evs': 2}
    min = 1.01
    max = 1000
    precision = 6

    def __init__(self, price, delimiter=r'/', default=None):
        if isinstance(price, basestring):
            if delimiter in price:
                splt = price.split(delimiter)
                self._set_value(float(splt[0]) / float(splt[1]) + 1.0)
            else:
                lookup = self.special_names.get(price.strip().lower())
                self._set_value(lookup or float(price))
        else:
            self._set_value(price)

    @property
    def rounded(self):
        if np.isnan(self._value):
            return self._value
        idx = (np.abs(self.steps - self._value)).argmin()
        return self.steps[idx]

    @classmethod
    def round(cls, vals):
        ret = np.zeros_like(vals)
        for i, v in enumerate(vals):
            ret[i] = cls.steps[np.argmin(np.abs(cls.steps - v))]
        ret[np.isnan(vals)] = np.nan
        return ret
