'''
UKHR CSV fieds: 
Meeting,Time,Race Type,Race Class,Title,Going,Furlongs,Prize,Minimum Age,Maximum Age,Mean Weight,Rating,Horse,Card Number,
Gender,Stall Number,Stall Percentage,Wearing,Days Since Last Run,Number of Results,Going Regression,Distance Regression,Age,
Weight (Stones & Pounds),Weight (Pounds),Total Wins,Recent Wins,Course Wins,Penalties,Allowances,Weight Delta,
Raw Rating,Jockey,Jockey Rating,Trainer,Trainer Rating,Trainer Form,Connections Rating,Form Trend,Form - Last Run,
Class,Winning Form,Speed,Alarms,Class Position,Win Class Probability,Win Class Probability (Normalised),Value Odds,
Value Odds (Probability),Forecast SP,Chase Jumping Ability,Ratings Position,CSV Version,Handicap,Value Place Odds,
BetFair SP Forecast Win Price,BetFair SP Forecast Place Price,Raw Adjusted for Age and Weight,Date,Elapsed Days,Systems,
Time 24 Hour,TJC Wins,TJC Runs,TJC PL,TJC SR,TJC ROI,TJC Type Wins,TJC Type Runs,TJC Type PL,TJC Type SR,TJC Type ROI,BST/GMT,
Runners,TC Wins,TC Runs,TC PL,TC SR,TC ROI,JC Wins,JC Runs,JC PL,JC SR,JC ROI,Value Odds (BetFair format),TJ Wins,TJ Runs,
TJ PL,TJ SR,TJ ROI,TRF_20RunsWins,TRF_20RunsRuns,TRF_20RunsPL,TRF_20RunsSR,TRF_20RunsROI,TRF_20RunsDays,TRF_2WeeksWins,
TRF_2WeeksRuns,TRF_2WeeksPL,TRF_2WeeksSR,TRF_2WeeksROI,TRF_4WeeksWins,TRF_4WeeksRuns,TRF_4WeeksPL,TRF_4WeeksSR,TRF_4WeeksROI,
Horse Form,Selling,Claiming,Auction,Novice,Maiden,Beginner,Hunter Chase,Raw Ranking,RAdj Ranking,Jockey Ranking,Trainer Ranking,
TrForm Ranking,Conn Ranking,Frm Ranking,Lst Ranking,Cls Ranking,WinF Ranking,Spd Ranking,HCP Ranking,Trainer Calendar Runs,
Trainer Calendar Wins,Trainer Calendar Return,Trainer Calendar P/L,Trainer Calendar SR%,Trainer Calendar ROI%,Trainer 5 year Runs,
Trainer 5 year Wins,Trainer 5 year Return,Trainer 5 year P/L,Trainer 5 year SR%,Trainer 5 year ROI%,
WRITE FAVOURITE RANKING,WRITE BETFAIR PLACED,WRITE BETFAIR PLACE S.P. HERE,WRITE BETFAIR S.P. HERE,WRITE GOING HERE,
WRITE IN RESULT HERE,WRITE IN S.P. HERE
'''

import codecs
import os
import csv
import collections
import numpy as np
import pandas as pd
import shutil

from utils import Folder, chunks, get_logger, timestamp, HOUR, DAY
from utils.world import POUND, STONE, FURLONG
from utils.errors import (ValidationError, NotFound, ParsingError, MultipleEntries)

from .datasource import AuthenticatedDataSource


UkhrRunLine = collections.namedtuple('UkhrRunLine', 'weight draw jockey ukhr_jcky_rtg trainer ukhr_trnr_rtg ukhr_est ukhr_rtg ukhr_value sex age result')
UkhrEventLine = collections.namedtuple('UkhrEventLine', 'obstacle race_class name going distance prize ukhr_race_rtg minimum_age maximum_age')

logger = get_logger(__name__)

NAME_ERRORS = dict((
    ('FIVEOCLOCK EXPRESS', 'FIVEOCLOCK EXRESS'),
))


class Ukhr(AuthenticatedDataSource):
    """ Class for accessing and buffering UKhorseracing data """
    url_base = 'https://www.ukhorseracing.co.uk'
    event_attributes = 'going', 'name', 'prize', 'distance', 'obstacle'
    runner_attributes = 'sex'
    run_attributes = 'ukhr_est', 'ukhr_value', 'ukhr_rtg', 'ukhr_jcky_rtg', 'ukhr_trnr_rtg', \
                    'weight', 'draw', 'result', 'jockey', 'trainer'
    earliest_record = float(timestamp("2013-01-01"))

    renamed_fields = {'JockeyRating': 'ukhr_jcky_rtg',
                      'TrainerRating': 'ukhr_trnr_rtg',
                      'Rating': 'ukhr_rtg',
                      'ValueOdds(Probability)': 'ukhr_value',
                      'StallNumber': 'draw',
                      'Gender': 'sex',
                      'ForecastSP': 'isp',
                      'BetFairSPForecastWinPrice': 'bsp_forecast',
                      'Time24Hour': 'time',
                      'Time': 'time12',
                      'Weight(Pounds)': 'weight',
                      'Weight(StonesPounds)': 'weight_stones_pound'}
    copied_fields = ('Jockey Trainer Date Result Meeting Furlongs Selling Claiming Auction Novice Maiden Beginner HunterChase '
                     'RaceClass RaceType Going MinAge MaxAge Title Horse Prize Age').split()
    _copy_list = renamed_fields.items() + [(v, v.lower()) for v in copied_fields]

    event_verification_checks = [('distance', lambda x: isinstance(x, float) and not np.isnan(x), 0.95),
                      ('obstacle', lambda x: isinstance(x, (str, unicode)) and (x == 'AW' or len(x) > 2), 0.95)]
    run_verification_checks = [('jockey', lambda x: isinstance(x, (str, unicode)) and len(x) > 4, 0.90),
                               ('trainer', lambda x: isinstance(x, (str, unicode)) and len(x) > 4, 0.90)]

    URL_LOGIN = '/membersProcess.asp'
    URL_MEMBERS = '/members.asp'
    URL_MEMBERS_DEFAULT = '/members/default.asp'
    URL_ARCHIVE = '/Archives/DisplayFolderContentsMembers.asp?Folder=%s'

    def _login(self):
        self.urlopen(self.URL_MEMBERS, cached=False, to_string=True)  # Get cookie first
        login_payload = dict(UKHRLoginIDField=self.username, UKHRPassword=self.password)
        self.urlopen(self.URL_LOGIN, method='POST', payload=login_payload, cached=False, to_string=True)

    def check_login(self, raise_errors=False):
        tree = self.urlopen(self.URL_MEMBERS_DEFAULT, cached=False, to_tree=True)
        for p in tree.xpath('//div[@id="content"]/p/text()'):
            if p.startswith('User ID and Password are not found'):
                return self._retraise(raise_errors, 'Account invalid')

        for form in tree.xpath('//div[@id="content"]/form/text()'):
            if 'Enter your Username' in form:
                return self._retraise(raise_errors, "Login failed")
        return True

    def check_login_offline(self, raise_errors=False):
        self.clear_cookies(expired_only=True)
        cookies = [c for c in self._cookies() if c.name.startswith('ASPSESSIONID')]
        return True if cookies else self._retraise(raise_errors, "Login cookie not found")

    def update(self, event, attribute=None, db=None):
        if event.course.country  not in self.available_countries:
            return
        if event.start_time < self.earliest_record:
            return

        if timestamp() < timestamp(event.start_time).daystart() - 2 * HOUR:
            # No data available yet. Should be somewhen between the last race and midnight
            return

        event_data, runner_data = self._race_data(event.start_time, event.course.name)
        for item in UkhrEventLine._fields[:7]:
            if not getattr(event, item) and getattr(event_data, item):
                try:
                    setattr(event, item, getattr(event_data, item))
                except ValidationError as e:
                    logger.warn("%s: %s", type(e).__name__, str(e))

        for run in event.runs:
            try:
                ukhr_line = runner_data[run.runner.name]
            except KeyError:
                continue
            else:
                for attr in ('ukhr_est', 'ukhr_rtg', 'ukhr_jcky_rtg', 'ukhr_trnr_rtg', 'ukhr_value'):
                    val = getattr(ukhr_line, attr)
                    if val:
                        setattr(run, attr, val)
                run.runner.sex = ukhr_line.sex

    def _daily_data(self, start_time, course_name):
        """ Find data for event, parse and/or download CSV if necessary """
        start_time = timestamp(start_time)
        try:
            daily_data = self.cache[float(start_time.daystart())]
        except KeyError:
            raise NotFound("Data for %s not in cache after update", start_time.format('int'))

        cache_key = self._generic_key(start_time, course_name)
        return daily_data, cache_key


    def match(self, fpath):
        """Match run_ids with data from UKHR"""
        runs = pd.read_csv(fpath)
        runs.columns = ['Run_ID', 'Start_time', 'Course_name', 'Horse_name']
        runs['horse_name'] = ''
        runs['date'] = ''
        for i in xrange(runs.shape[0]):
            start_time = runs.Start_time.values[i]
            course_name = runs.Course_name.values[i]
            if float(timestamp(start_time).daystart()) in self.cache:
                daily_data, cache_key = self._daily_data(start_time, course_name)
                if cache_key in daily_data:
                    event_data, runner_data = daily_data[cache_key]
                    horse_name = runs.Horse_name.values[i].title()
                    if horse_name in runner_data.keys():
                        runs.set_value(i, 'horse_name', horse_name)
                        runs.set_value(i, 'date', timestamp(start_time).format('%Y-%m-%d'))
        runs.to_csv("../run_id_match_V1.csv")

    def _read_csv(self, fpath, raise_errors=False):
        def line_repair(iterable):
            """ Fixes some quoting errors on the fly """
            for line in iterable:
                if line != '\n':
                    yield line.replace(' "",', '",')

        parsed_data = dict()
        with codecs.open(fpath, 'rb', 'ascii', errors='ignore') as f:
            reader = csv.DictReader(line_repair(f), delimiter=',', quotechar='"', doublequote=True)
            # Older files used headers like "Weight (Stones & Pounds)"
            reader.fieldnames = self._prepare_csv_fieldnames(reader.fieldnames)
            trim_reader = self._line_trimmer(reader)

            for ev_chunk in chunks(trim_reader, lambda x: (x['meeting'], x['time'])):
                try:
                    day_stamp, cache_key, ev_info, runs_data = self._read_event_chunk(ev_chunk)
                except ParsingError as pe:
                    if raise_errors:
                        raise
                    else:
                        logger.error("%s -- Event skipped", pe)
                else:
                    daily_data = parsed_data.setdefault(day_stamp, dict())
                    daily_data[cache_key] = ev_info, runs_data
        return parsed_data

    def _prepare_csv_fieldnames(self, fieldnames):
        fieldnames = [key.strip().replace(' ', '').replace('&', '') for key in fieldnames]
        replacements = dict(MinimumAge='MinAge', MaximumAge='MaxAge')
        return [replacements.get(field, field) for field in fieldnames]

    def _get_value(self, line, key):
        try:
            val = line[key]
        except KeyError:
            return None

        if val in ('', '-', None):
            return

        # There were some bogus fields containing {} in some old files
        val = val.split('{', 1)[0].strip()

        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass

        return val

    def _line_trimmer(self, reader):
        for line in reader:
            yield {outkey: self._get_value(line, inkey) for inkey, outkey in self._copy_list}

    def _read_event_chunk(self, chunk):
        day_stamp, ev_key, ev_info = self._parse_event_info(chunk[0])
        runs_data = self._parse_event_runners(chunk)
        return day_stamp, ev_key, ev_info, runs_data

    def _parse_event_info(self, line):
        try:
            event_time = timestamp(line['date'] + line['time'], fmt="%d/%m/%Y%H:%M", tz=self.tz)
        except (TypeError, ValueError):
            if not line['date'] or not '/' in line['date']:
                raise ParsingError("Date Error: '%s', Line: %s", line['date'], line)

            try:
                event_time = timestamp(line['date'] + line['time12'].rjust(5, '0'),
                                       fmt="%d/%m/%Y%H:%M", tz=self.tz)
            except (TypeError, ValueError):
                raise ParsingError("Time error: '%s' / '%s', Line: %s", line['time'], line['time12'], line)
            if event_time - event_time.daystart() < 11 * HOUR:
                event_time += 12 * HOUR
        try:
            distance = line['furlongs'] * FURLONG
        except (TypeError, ValueError):
            distance = None

        prize = line['prize']
        if isinstance(prize, str):
            # Previous conversion attempts failed, probably due to currency chars
            # TODO: Handle Sterling / Euro values correctly
            # if course.country == 'IRL':
            #     prize = self.converter.convert(prize, 'EUR', 'GBP', timestamp=event_time)
            try:
                prize = float(''.join(c for c in prize if c in '0123456789.'))
            except (TypeError, ValueError):
                prize = None


        for key in ('selling', 'claiming', 'auction', 'novice', 'maiden', 'beginner', 'hunterchase'):
            # No need to strip the result, already done before
            val = line[key]
            if val:
                race_class = val
                break
        else:
            race_class = None

        try:
            race_rtg = float(line['raceclass'])
        except (TypeError, ValueError):
            race_rtg = None

        # 'obstacle race_class name going distance prize minimum_age maximum_age'
        ev_record = UkhrEventLine(line['racetype'], race_class, line['title'],
                                  line['going'], distance,
                                  prize, race_rtg,
                                  line['minage'], line['maxage'])

        course_name = line['meeting']
        cache_key = self._generic_key(event_time, course_name)
        return float(event_time.daystart()), cache_key, ev_record

    def _parse_event_runners(self, evdata):
        runners = dict()
        for line in evdata:
            if not line['horse']:
                continue
            try:
                runner_name = line['horse']
            except ValueError:
                continue

            ukhr_est = line['bsp_forecast']
            if not ukhr_est:
                # Contains zeros, make sure to replace them
                ukhr_est = None
            ukhr_rtg = line['ukhr_rtg']
            ukhr_rtg = max(float(ukhr_rtg), 0) if ukhr_rtg else None

            try:
                ukhr_value = 1 / float(line['ukhr_value'])
            except (TypeError, ValueError):
                ukhr_value = None

            weight = None
            try:
                weight = float(line['weight']) * POUND
            except (TypeError, ValueError):
                wsp = line['weight_stones_pound']
                if wsp:
                    splt = wsp.strip().split('-', 1)
                    try:
                        weight = int(splt[0]) * STONE + int(splt[1]) * POUND
                    except (IndexError, ValueError):
                        if line['weight']:
                            logger.warn("Weight Error: '%s', Line: %s", line['weight'], line)

            age = line['age']
            age = age if 2 <= age < 20 else None

            runners[runner_name] = UkhrRunLine(weight, line['draw'], line['jockey'], line['ukhr_jcky_rtg'],
                                               line['trainer'], line['ukhr_trnr_rtg'], ukhr_est, ukhr_rtg, ukhr_value,
                                               line['sex'], age, line['result'])
        return runners

    def update_archive(self):
        """ Update the stored archive for every single day since the earliest available record """
        ts = timestamp(self.earliest_record)
        now = timestamp()

        while ts < now:
            if ts == ts.monthstart():
                logger.info("Verifying / fetching %", ts.format("%Y-%m"))
            try:
                self._update_cache(ts)
            except Exception:
                logger.error("Verifying / fetching % failed", ts.format("%Y-%m-%d"))
                #logger.exception()
            ts += DAY

    def _update_cache(self, start_time):
        """ Download csv if local file does not exist or is outdated """

        stamp = timestamp(start_time)

        try:
            fpath = self._cached_csv(start_time)
        except NotFound:
            fpath = self._download_csv(start_time)

        self.cache.update(self._read_csv(fpath))

    def _cached_csv(self, start_time):
        fpath = self._csv_filename(start_time)
        if os.path.isfile(fpath):
            return fpath

        # Lookup for downloads with original UKHR filename
        date_str = timestamp(start_time).format('%Y%m%d')
        yeardir = Folder(fpath)
        fnames = [fname for fname in yeardir.listdir() if fname.startswith(date_str)]
        if not fnames:
            raise NotFound("No cached CSV files found for that date")

        if len(fnames) > 1:
            # Be a bit more specific and pick the orignal summaries
            fnames = [fname for fname in fnames if 'summary' in fname]
            if len(fnames) > 1:
                raise MultipleEntries("Too many files found (%s)", ', '.join(fnames))

        fname = fnames[0]
        found_fpath = yeardir.join(fname)

        # Rename the file, so that we can grab it right away next time
        shutil.move(found_fpath, fpath)
        return fpath

    def _download_csv(self, start_time=None):
        self.assert_login()
        dt = timestamp(start_time)
        if dt >= timestamp().daystart():
            url = self._recent_csv_url()
            if dt.format('%Y%m%d') not in url:
                raise NotFound("CSV for this date not yet available")
        else:
            url = self._archived_csv_url(dt)

        resp = self.urlopen(url, cached=False, to_string=False)
        self._verify_csv_header(resp.content)

        fpath = self._csv_filename(dt)
        resp_str = resp.content.strip()
        with open(fpath, 'w') as csv_file:
            csv_file.write(resp_str)
        return fpath

    def _recent_csv_url(self):
        """ All CSVs are also instantly available in the archive. So this
            serves only as shortcut or fallback, and could also be removed for simplicity
        """
        tree = self.urlopen(self.URL_MEMBERS_DEFAULT, to_tree=True)
        for url in tree.xpath("//div[@id='sidebar']//a/@href"):
            if url.endswith('.csv') and 'summary' in url:
                break
        else:
            raise ParsingError('Keinen CSV-Link auf der Downloadseite gefunden')
        return url

    def _archived_csv_url(self, start_time):
        dt = timestamp(start_time)
        tree = self.urlopen(self.URL_ARCHIVE % dt.year, to_tree=True)
        csv_links = tree.xpath("//div[@id='content']/p/a[contains(@href, '.csv')]/@href")
        search = '/' + dt.format("%Y%m%d")
        filtered_links = [l for l in csv_links if search in l]
        if not filtered_links:
            raise NotFound("CSV for %s not found (yet) in archives", dt.format('int'))
        if len(filtered_links) != 1:
            raise ParsingError("Found more than one daily download link")
        url = filtered_links[0]

        if not url.startswith('http') and not url.startswith('/'):
            # It usually should be a relative link, so we ned to prefix it
            url = self.URL_ARCHIVE.rsplit('/', 1)[0] + '/' + url
        return url

    def _verify_csv_header(self, csv_content):
        if '<html' in csv_content[:1000]:
            raise ParsingError("File seems to contain HTML content")
        for header_item in ("Meeting", "Title", "Going", "Rating"):
            if header_item not in csv_content[:1000]:
                raise ParsingError("No valid CSV header found")
        if len(csv_content) < 10000:
            raise ParsingError("CSV unusually small")
        return True

    def _csv_filename(self, start_time, monthly=False):
        """ This creates a unique filename per day. Does not correspond to the filename served from UKHR """
        base = 'ukhr_'
        if not monthly:
            base += 'daily_'
        ts = timestamp(start_time)
        ts_start = ts.monthstart() if monthly else ts.daystart()
        date_str = ts_start.format('int')
        year_dir = self.path.join(str(ts.year))
        if not os.path.isdir(year_dir):
            os.mkdir(year_dir)
        return self.path[str(ts.year)].join(base + date_str + '.csv')
