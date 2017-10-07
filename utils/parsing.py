# -*- coding: utf-8 -*-
""" Collection of small functions for parsing common horseracing input.
    No function shall throw an error for common issues and just return 
    None or another empty expected result instead.
    
    Parsing functions, that only cover special issues of single datasources
    shall remain in their modules.
"""

import string
import re
from collections import namedtuple

from .times import timestamp, YEAR
from .helpers import mklist, mkset, treewalk  # @UnusedImport for reimport
from .world import (Country, Obstacle, Handicap, RaceClass, Distance, Going,
                    HORSE_LENGTH, NECK, SHORT_NECK, HEAD, SHORT_HEAD, NOSE)
from .names import Horsename
from .database import Run, EquipmentList


ParsedDescription = namedtuple('ParsedDescription', 'handicap obstacle race_class distance date_of_birth days_idx')

UNICODE_TABLE = {
    ord(u'ä'): u'ae',
    ord(u'Ä'): u'Ae',
    ord(u'å'): u'a',
    ord(u'Å'): u'A',
    ord(u'ö'): u'oe',
    ord(u'Ö'): u'Oe',
    ord(u'ü'): u'ue',
    ord(u'Ü'): u'Ue',
    ord(u'ß'): u'ss',
}

COURSES_TRASH = frozenset(('Hotpots', 'Oaks', 'Without', 'Derby', 'Doubles', 'Special', 'Champion',
        'Stewards', 'Fri', 'Tues', 'Wed', 'Thurs', 'All', 'Saturday', 'Daily', 'Match', 'Festival',
        'Breeders', 'Breeding', 'Shergar', 'InterDom', 'QE', 'BHA', 'BHB', 'Winning', 'Jockey', 'Guineas',
        'Jackpot', 'Test', 'Triple', 'Betfair', 'Top', 'First', 'Will', 'Aus', 'KG', '200m', 'Daily',
        'Scoop6', 'Stewards', 'Spezialwetten', 'No', 'Breeding & Bloodstock'))

EVENTS_TRASH = frozenset(('TO BE PLACED', 'Forecast', 'Without Fav', 'Without Fav(s)', 'Win', 'Winning Stall',
                          'Accumulator', 'Reverse FC', 'Name The Fav', 'NTF', 'WITHOUT FAV', 'Without fav',
                          'Name the Fav', 'Odd v Even', 'To Be Placed', 'Keep The Race?', 'Winning Distance (Odds)',
                          'Longest Winning SP', 'Jockey Challenge', 'ISP % Return', 'Irish Trained Winner?',
                          'No of Finishers', 'Weight of Winner', 'Will They Finish?', 'Jockey CHallenge',
                          'Jockey challenge'))

COUNTRY_TRASH = frozenset(('ANTEPOST', 'Test Aus Event', 'World Aquatics Championships 2009', '', 'Breeding & Bloodstock'))

DESC_TRASH = frozenset(('ANTEPOST', '(AvB)', '(W/O', '(Hcap)', 'Dist', 'Special', '(Group)', '(Stall)', 'Scoop',
                        'Match Bets', '(NTF)', '(RFC)', '(F/C)', '(OvE)', '(T/J)', '(1&2)', '(1/2)'
                        'Breeding', 'Accumulator', 'Antepost', 'Will Racing Go Ahead?', 'Quinella',
                        'Winning Margin'))

NAME_TRASH = frozenset(('YES', 'NO', '2 WINNERS OR MORE'))

MARKET_TRASH = frozenset((102817643,))



def trash_scan(course=None, country=None, event=None, description=None, name=None, selections=None, path=None,
               market=None):
    if event and (event in EVENTS_TRASH or event.endswith('TBP') or event.startswith('Winners') or
                  event.endswith('Quinella') or event.endswith('Exacta')):
        return True
    if course and (course in COURSES_TRASH or course.startswith('Breeders') or re.findall('(\d+(?:st|nd|rd|th))', course)):
        return True
    if name and name.upper() in NAME_TRASH:
        return True
    if market in MARKET_TRASH:
        return True
    if selections:
        selections = mkset(item.upper() for item in mklist(selections))
        if NAME_TRASH & selections:
            return True
    if path:
        if '(' in path:
            comment = path[path.find('(') + 1:path.find(')')]
            try:
                Country(comment).code
            except ValueError:
                return True
        # Same as for description basically
        for trash in DESC_TRASH:
            if trash in path:
                return True
    if description:
        for trash in DESC_TRASH:
            if trash in description:
                return True
    if country and country in COUNTRY_TRASH:
        return True
    return False


def description_parser(desc, race_start=None):
    """
    We have to handle a lot of implied properties, so we have to carefully go step by step
    from the most uncertain property setting to the more reliable ones and allow overwriting
    of the previously found settings. We do not pass any default value, so we don't
    confuse any later data merging with just guessed values
    
    Information tried do determine in the following order:
    1. Race index of the day (US only)
    2. Distance
    3. Estimate of birth day of the horses
    4. Race class
    5. Obstacle type
    6. Handicapping type
    """

    desc = desc.strip().upper()
    desc_splt = desc.split()
    days_idx = distance = years = date_of_birth = race_class = obstacle = handicap = None

    if desc[0] == 'R':
        if desc.startswith('RACE'):
            try:
                days_idx = int(desc_splt[1])
            except ValueError:
                pass
            else:
                del desc_splt[:2]
        elif desc[1] in '123456789':
            try:
                days_idx = int(desc_splt[0][1:])
            except ValueError:
                pass
            else:
                del desc_splt[0]

    if desc_splt:
        try:
            distance = Distance(desc_splt[0]).meters
        except ValueError:
            pass
        else:
            del desc_splt[0]

    if race_start and desc_splt:
        for i in range(len(desc_splt)):
            item = desc_splt[i]
            if item[0] in '23456' and item[1:3] == 'YO':
                del desc_splt[i]
                if item[-1] == '+':
                    # Also elder horses are admitted, so we can not guess the horses age
                    break
                years = int(item[0])
                date_of_birth = (timestamp(race_start) - YEAR * years).yearstart()
                break

    # After this point, we don't need to delete parts from the split anymore and can directly iterate
    for item in desc_splt:
        # Fishing for race class related stuff
        if item in ('CLASS', 'LISTED'):
            race_class = RaceClass.LISTED
            handicap = Handicap.CONDITIONS
        elif item.startswith('GR'):
            if len(item) == 4 and item[3] in '123':
                race_class = 'G' + item[3]
                obstacle = Obstacle.FLAT if item[2] == 'P' else Obstacle.CHASE
                # Group or grade? P or D?
                handicap = Handicap.WEIGHT_FOR_AGE if int(item[3]) == 1 else Handicap.CONDITIONS
            elif 'GRAND NATIONAL' in desc:
                race_class = RaceClass.GRADE1
                obstacle = Obstacle.CHASE
                handicap = Handicap.WEIGHT_FOR_AGE
        elif item.startswith('CLAIM'):
            race_class = RaceClass.CLAIMING
        elif item.startswith('SELL'):
            race_class = RaceClass.SELLING
        elif item in ('INHF', 'NHF'):
            race_class = RaceClass.BUMPER
            obstacle = Obstacle.FLAT
        elif item == 'NURSERY':
            race_class = RaceClass.BUMPER
            obstacle = Obstacle.FLAT
            handicap = Handicap.HANDICAP_USUAL
            years = 2

    for item in desc_splt:
        # Fishing for obstacle and handicapping stuff
        if item in ('STKS', 'COND'):
            handicap = Handicap.CONDITIONS
        elif item == 'HCAP':
            handicap = Handicap.HANDICAP_USUAL
        elif item == 'HRD':
            obstacle = Obstacle.HURDLE
        elif item == 'CHS':
            obstacle = Obstacle.CHASE
        elif item == 'FLAT':
            obstacle = Obstacle.FLAT

    if years == 2 and not obstacle:
        obstacle = Obstacle.FLAT

    return ParsedDescription(handicap, obstacle, race_class, distance, date_of_birth, days_idx)


def racecard_name_parser(rc_entry, default_country=None, error_country=None):
    rc_entry = rc_entry.strip()
    if rc_entry:
        name = rc_entry
    else:
        raise ValueError('Empty horse name supplied')
    cardno = None
    if name[0] in '0123456789':
        if not '.' in name:
            # Probably malformed name, skipping numbers
            name = name.split(None, 1)[1]
            cardno = None
        else:
            cardno, name = name.split('.', 1)
            try:
                cardno = int(cardno)
            except ValueError:
                try:
                    # Convert to str, as it might be unicode, which does not
                    # support translate with two arguments
                    cardno = int(str(cardno).translate(None, 'ABCD_ '))
                except ValueError:
                    cardno = None
    else:
        cardno = None

    rc_country = None
    if '(' in name:
        name, rc_country = name.split('(')[:2]
        rc_country = rc_country.strip('() ').upper()

    if not rc_country and default_country:
        country = Country(default_country).code
    elif rc_country:
        try:
            country = Country(rc_country).code
        except ValueError:
            country = error_country
    else:
        country = error_country
    name = Horsename.validated(name)
    return name, country, cardno


def name_country_rating_split(entry):
    """ Split things like: MANDURO (GER) 135p """
    entry = entry.strip("() ")  # Avoids empty terminal part on re.split
    if not entry:
        return None, None, None
    entry_pts = [pt for pt in re.split('\W+', entry) if pt]
    rating = None
    rtg = entry_pts[-1]
    if len(rtg) >= 2 and rtg[0] in string.digits and rtg[1] in string.digits:
        rating = parse_int(rtg, min=0, max=1000)

    if rating is not None:
        entry = entry.rsplit(None, 1)[0]
        entry_pts = entry_pts[:-1]

    if '(' in entry:
        country = Country.validated(entry_pts[-1])
        entry = entry.split('(', 1)[0].strip()
    else:
        country = None

    name = Horsename.validated(entry)
    return name, country, rating


def parse_equipment(eq):
    eq = eq.strip('() \n').replace(' ', '')
    if len(eq) > 10:
        raise ValueError("Invalid equipment string: %s" % eq)
    if not eq or '-' in eq:
        return None
    eq_list = []
    exceptions = {'cp': 'cp', 'es': 'es', 'e/s': 'es'}
    for excp, val in exceptions.items():
        if excp in eq:
            eq_list.append(val)
            eq = eq.replace(excp, ' ')

    eq_list += [c for c in eq.split('+') if c in string.ascii_letters]
    return EquipmentList(sorted(eq_list))


def parse_unicode(ustr):
    return str(ustr.decode('utf8').translate(UNICODE_TABLE))


def parse_weight(weight):
    # We have some values '9' instead of '9-0'
    if 0 < len(weight) < 3 and '-' not in weight:
        weight += '-0'
    weight = weight.split('(', 1)[0]  # Fix: 8-2 (8-1)
    if weight and weight not in ('0-0', '-'):
        return weight


def parse_forecast(fc):
    # Betting Forecast : 13/8 My Tent Or Yours, 2/1 Vroum Vroum Mag, 7/2 Identity Thief
    if not fc or not ':' in fc:
        return dict()
    fc = fc.split(':', 1)[1]
    fc_dict = dict(tuple(reversed(x.strip().split(None, 1))) for x in fc.upper().split(','))
    if all(val == '1000' for val in fc_dict.values()):
        return dict()
    return fc_dict


def parse_win_time(wt_str):
    # 1m 37.9s
    wt = wt_str.strip(' s')
    if not wt:
        return None
    elif 'm' in wt:
        splt = wt.split('m')
        return float(splt[0]) * 60 + float(splt[1])
    else:
        return float(wt)


def parse_bl(bl):
    if not bl:
        return None
    bl_factor_chars = {u'¼': 0.25, u'½': 0.5, u'¾': 0.75}
    bl_dict = {'': 0, 'DS': 30 * HORSE_LENGTH, 'DIST': 30 * HORSE_LENGTH, 'NK': NECK, 'SNK': SHORT_NECK,
               'HD': HEAD, 'SHD': SHORT_HEAD, 'SH': SHORT_HEAD, 'NSE': NOSE, 'NS': NOSE, 'DH': 0,
               'DIS': None, 'DQ': None, 'A': None, None: None}
    bl = bl.upper()
    if bl in bl_dict:
        return bl_dict[bl]
    elif bl.startswith('-'):
        # Negative entries, what to do with them best?
        # -1 lately found in /horse-racing/190212/Gulfstream_Park-US-GP/1235
        # -1 was caused by disqualification, horse was placed by referee 3rd instead of 2nd
        # -1 could have been just ignored in that case, still the best guess
        return 0
    elif bl.count('/') == 1:
        counter, denominator = bl.split('/')
        return float(counter) / float(denominator) * HORSE_LENGTH
    else:
        for char, val in bl_factor_chars.items():
            if bl.endswith(char):
                return (int(bl.strip(char) or 0) + val) * HORSE_LENGTH
        else:
            return float(bl) * HORSE_LENGTH


def parse_int(val, min=None, max=None):
    if val is None:
        return
    if isinstance(val, basestring):
        val = ''.join(c for c in val if c in string.digits)
    try:
        val = int(val)
    except ValueError:
        return None
    if min is not None and val < min:
        return None
    if max is not None and val > max:
        return None
    return val


def parse_result(result):
    try:
        return int(result)
    except (TypeError, ValueError):
        result = result.strip().upper()
        if not result:
            return None
        if result in 'CUFPR' or result in ('PU', 'UR', 'RR', 'RTR', 'HR', 'RO', 'SU', 'BD', 'CO'):
            # pulled up, unseated rider, refused race, refused to race, hit rails, ran out, slipped up, ... , ...
            return Run.NOT_FINISHED
        elif result in ('DQ', 'DIS'):
            return Run.DISQUALIFIED
        else:
            raise ValueError("Unknown race result: %s" % result)


def parse_color_gender(line):
    """ Split lines like: b h, chg """
    line = line.strip()
    if not line:
        return None, None
    if 'or' in line:
        line = line.split('or')[-1]
    pts = line.split()
    if len(pts) == 1 and pts[0][-1] in 'cfmhg':
        return pts[0][:-1], pts[0][-1]
    elif len(pts) == 2:
        return tuple(pts)
    else:
        raise ValueError("Invalid color and gender string: %s" % line)


def parse_going(going_str, obstacle=None):
    track_replacements = {'HURDLES': 'HURDLE', 'NH': 'FLAT', 'SPRINT': 'FLAT', 'COUNTRY': 'FLAT', 'X-COUNTRY': 'FLAT'}
    going_str = going_str.upper()
    if not ':' in going_str:
        going = going_str
    elif not any(item in going_str for item in ('FIRST', 'RACE')):
        going_str = going_str.replace(' COURSE:', ':')
        goings = dict()
        for going_sub_groups in re.findall(r'([\w-]+:[^:]+)($|\s)', going_str):
            going_sub_str = going_sub_groups[0]
            track, going = going_sub_str.strip().split(':', 1)
            track = track_replacements.get(track, track)
            goings[track] = going.strip()
        if not obstacle:
            track = 'FLAT'
        else:
            track = Obstacle.readable[obstacle].upper()

        for key in (track, 'ALL-WEATHER', 'TURF', 'DIRT'):
            going = goings.get(key, '')
            if going or track != 'FLAT':
                break
        else:
            return None
    else:
        going = going_str.split(':')[-1]

    going = going.rstrip(',;')
    if ',' in going and not '(' in going:
        going = going.split(',', 1)[1]
    going = going.split('(', 1)[0].strip()
    if 'ALL-WEATHER' in going:
        if going.startswith('ALL-WEATHER') and len(going) < 15:
            return Going.ALL_WEATHER
        else:
            going = going[11:].strip()
    if going in ('TURF', 'DIRT'):
        return None
    if going.startswith(('TURF', 'DIRT')):
        going = going[4:].strip()

    if going.startswith('STANDARD') and '/' in going:
        going = 'STANDARD TO ' + going.split('/', 1)[1].strip()

    return going or None


def read_xpath(tree, path, encoding='ascii', errors='ignore'):
    val = tree.xpath(path)
    if isinstance(val, list):
        val = ' '.join(v.xpath('normalize-space(.)') for v in val)
    if isinstance(val, unicode) and encoding:
        val = val.encode(encoding, errors=errors)
    if not isinstance(val, basestring):
        raise ValueError("Unexpected element type: %s %s" % (val, type(val)))
    return val
