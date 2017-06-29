'''
General definition of a datasource for horse racing.
'''
import string

from utils import Cache, Folder, get_logger, timestamp, HOUR, MINUTE
from utils.database import EventConsistencyError, PersistentCookieJar
from utils.errors import (ParsingError, ConnectionError, ValidationError,
                          RetriesExceeded, NotFound, NotAvailable)
from utils.gutils.crawler import XMLCrawler, DEFAULT_HEADERS
from utils.gutils import RLock, Timeout, with_timeout

logger = get_logger(__package__)


class EventUpdater(object):
    """ Mainly abstract class grouping all logic for safely updating events """

    available_countries = 'GBR', 'IRL'
    earliest_record = None
    availability_min = 0.8
    full_update_available = False

    # Define the attributes, which the updater can deliver
    event_attributes = None
    runner_attributes = None
    run_attributes = None

    def update(self, event, attribute=None, db=None):
        """ All specified attributes of the event shall be updated. If attribute 
            is not specified, update all attributes.
        """
        raise NotImplementedError

    def update_if_required(self, event, attribute=None, db=None, timeout=None):
        """ Try to find one empty run attribute. Skip unnecessary updates """
        if event.runs and self.run_attributes:
            for attr in self.run_attributes:
                attrs = tuple(getattr(run, attr) for run in event.runs)
                if not all(attrs):
                    break
            else:
                # Went through all attrs, none breaked, so no update is required
                return
        self.update_failsafe(event, attribute=attribute, db=db, timeout=timeout)

    def update_failsafe(self, event, attribute=None, db=None, mute=None, timeout=None):
        if timeout is not None:
            return with_timeout(timeout, self._update_failsafe, event, attribute=attribute, db=db, mute=mute)
        else:
            return self._update_failsafe(event, attribute=attribute, db=db, mute=mute)

    def _update_failsafe(self, event, attribute=None, db=None, mute=None):
        if mute is None:
            mute = set()
        runs_count = len(event.runs)
        try:
            self.update(event, attribute=attribute, db=db)
        except (ConnectionError, ParsingError, EventConsistencyError, ValidationError,
                RetriesExceeded, NotFound, NotAvailable, Timeout) as e:
            if type(e) not in mute and not event.abandoned:
                logger.warn("%d|%s: %s update: %s %s", event.id, event.short_description, type(self).__name__,
                            type(e).__name__, e)
            return
        except Exception as e:
            logger.exception("%d|%s: %s update: %s %s", event.id, event.short_description, type(self).__name__,
                             type(e).__name__, e)
            return

        new_runs_count = len(event.runs)
        if new_runs_count and (runs_count == new_runs_count or not runs_count):
            logger.debug("%d|%s: %s update: Finished", event.id, event.short_description, type(self).__name__)
        elif not new_runs_count:
            logger.debug("%d|%s: %s update: No runs created", event.id, event.short_description,
                         type(self).__name__)
        else:
            logger.info("%d|%s: %s update: %d runs added", event.id, event.short_description,
                        type(self).__name__, new_runs_count - runs_count)

    @classmethod
    def actual_coverage(cls, runs):
        """ Returns the percentage of runs, for which data was actually found with this datasource """
        if not cls.run_attributes:
            return 0
        covered = [run for run in runs if cls.covers_run(run)]
        if not len(covered):
            return 1
        available = sum(1 for run in covered if getattr(run, cls.run_attributes[0], None) is not None)
        return float(available) / len(covered)

    @classmethod
    def covers_run(cls, run):
        """ Determines, if a the datasource should be able to provide information for this run """
        if run.event.course.country not in cls.available_countries:
            return False
        if cls.earliest_record and run.event.start_time < cls.earliest_record:
            return False
        return True


class DataSource(EventUpdater):
    """
    All implemented data sources should be a subclass of this and use the
    already offered mechanisms for accessing the internet:
    """
    tz = 'Europe/London'  # The timezone, in which all dates of the source are specified
    concurrency = 5

    def __init__(self, basepath=None, cookiejar=None, **kwargs):
        self.path = Folder(basepath).getfolder(type(self).__name__.lower()) if basepath else None
        self.crawler = self._init_crawler(cookiejar=cookiejar)
        self._cache = Cache(100, HOUR)

    def _init_crawler(self, cookiejar=None, ssl_options=None, concurrency=None):
        return XMLCrawler(concurrency=concurrency or self.concurrency,
                          headers=DEFAULT_HEADERS,
                          cookiejar=cookiejar if cookiejar is not None else PersistentCookieJar(),
                          network_timeout=30,
                          connection_timeout=5,
                          max_retries=2,
                          ssl_options=ssl_options)

    def available(self):
        """ Check availability of the data source by requesting the main page """
        try:
            return self.urlopen('/', to_string=True) != ''
        except Exception as e:
            logger.exception("Availability check failed with %s", type(e).__name__)
            return False

    def urlopen(self, url, **kwargs):
        """ Provide a simplified interface for retrieving cached web pages """
        url = self._prefix_url(url)
        return self.crawler.urlopen(url, **kwargs)

    def _prefix_url(self, url):
        """ Add the domain to a link, in case the link is relative """
        if url.startswith('http') or not hasattr(self, 'url_base'):
            # Don't touch absolute URLs
            return url
        if not self.url_base.endswith('/') and not url.startswith('/'):
            return self.url_base + '/' + url
        else:
            return self.url_base + url

    def _generic_key(self, start_time, course_name):
        """ Create a unique key, mainly for finding a race within the daily dictionaries """
        return int(timestamp(start_time, tz=self.tz)), self._course_key(course_name)

    def _format_generic_key(self, key):
        """ Print a generic key e.g. for error messages """
        return timestamp(key[0]).format('%H:%M ') + key[1].capitalize()

    def _course_key(self, course_name):
        # Trim country tags
        course_name = course_name.lower().split('(', 1)[0]
        if course_name.startswith("the "):
            course_name = course_name[4:]
        # Keep out any bullshit from the names
        course_name = ''.join(c for c in course_name if c in string.ascii_lowercase)
        return course_name[:5]

    def kill(self):
        """ Play along with other greenlet containers """
        self.crawler.kill()


class AuthenticatedDataSource(DataSource):
    """ Provides additional helpers for login procedures """
    login_check_interval = HOUR

    def __init__(self, username=None, password=None, **kwargs):
        super(AuthenticatedDataSource, self).__init__(**kwargs)
        self.username = username
        self.password = password
        self._login_lock = RLock()
        self._login_assert_lock = RLock()
        self._login_status = False
        self._login_last_check = 0

    def login(self):
        """ Try to login and verify it afterwards. Make sure only one
            greenlet or thread tries this at the same time.
        """
        with self._login_lock:
            self._login()
            self.check_login(raise_errors=True)
            self.check_login_offline(raise_errors=True)
            self._login_status = True
            self._login_last_check = float(timestamp())

    def _login(self):
        """ Do the actual login """
        raise NotImplementedError("Should be overridden by subclasses")

    def assert_login(self):
        """ Check if we're logged in, and trigger a login, if necessary """
        with self._login_assert_lock:
            # This makes sure, that not multiple processes try to login concurrently,
            # when the source was offline or the program just started up
            if self.login_verified:
                return
            self.login()

    def check_login(self, raise_errors=False):
        """ Actively request a page and verify the successful login """
        raise NotImplementedError("Should be overridden by subclasses")

    def check_login_offline(self, raise_errors=False):
        """ Check local state if we got a valid login. This is not always 
            possible. So this check is optional and by default should
            return True.
        """
        return True

    def _retraise(self, raise_errors, reason, *args, **kwargs):
        """ Simple helper for either returning False or raising the appropriate error """
        __tracebackhide__ = True
        error_type = kwargs.pop('error_type', ConnectionError)
        if args:
            reason = reason % args
        if raise_errors:
            raise error_type(reason)
        return False

    @property
    def login_verified(self):
        """ Make sure we actively check the login state on a regular basis.
            If the login was verified, the state is kept for a while without
            further rechecks.
        """
        now = timestamp()
        recheck_interval = self.login_check_interval if self._login_status else MINUTE
        if now - self._login_last_check < recheck_interval:
            online_check = self._login_status
        else:
            online_check = self.check_login()
        status = online_check and self.check_login_offline()
        self._login_status = status
        self._login_last_check = float(now)
        return status

    def clear_cookies(self, expired_only=False):
        now = timestamp()
        for c in self._cookies():
            if not expired_only or (c.expires and now > c.expires):
                del self.crawler.cookiejar[c.name]

    def _cookies(self):
        """ List all cookies relevant for the datasource """
        domain = self.url_base.split('.')[-2]
        return [c for c in self.crawler.cookiejar if domain in c.domain]
