from __future__ import absolute_import

import os
import string
import copy
import StringIO
import logging

from logging import StreamHandler, Logger
from logging.handlers import HTTPHandler, SMTPHandler, TimedRotatingFileHandler


def get_logger(name='', level=logging.INFO, fmt=logging.BASIC_FORMAT, handler=None, colored=False):
    name = name.split('.')[-1] if name is not None else '__main__'
    logger = logging.getLogger(name)
    logger.propagate = False
    if level is not None:
        logger.setLevel(level)
    if not len(logger.handlers):
        if handler is None:
            handler = logging.StreamHandler()
            if colored:
                handler.setFormatter(ColoredFormatter(fmt))
            else:
                handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

logger = get_logger(__package__)


class LogPrefixer(Logger):
    """ Adds a constant prefix to a given logger """

    def __init__(self, logger, prefix):
        assert isinstance(logger, Logger)
        assert isinstance(prefix, str)
        self.logger = logger
        self.prefix = prefix

    @property
    def level(self):
        return self.logger.level

    @property
    def parent(self):
        return self.logger.parent

    def setLevel(self, level):
        self.logger.setLevel(level)

    def _log(self, level, msg, args, exc_info=None, extra=None):
        msg = self.prefix + msg
        return self.logger._log(level, msg, args, exc_info=exc_info, extra=extra)

    def handle(self, record):
        self.logger.handle(record)

    def addHandler(self, hdlr):
        return

    def removeHandler(self, hdlr):
        return


class ColoredFormatter(logging.Formatter):
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    # The background is set with 40 plus the number of the color, and the foreground with 30
    # These are the sequences need to get colored ouput
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        'WARNING': YELLOW,
        'INFO': GREEN,
        'DEBUG': BLUE,
        'CRITICAL': MAGENTA,
        'ERROR': RED
    }

    @classmethod
    def colorize(cls, text, color=RED):
        if isinstance(color, basestring):
            color = getattr(cls, color.upper(), 0)
        return cls.COLOR_SEQ % (30 + color) + text + cls.RESET_SEQ

    def format(self, record):  # @ReservedAssignment
        levelname = record.levelname
        if levelname in self.COLORS:
            record = copy.copy(record)
            record.levelname = self.colorize(levelname, self.COLORS[levelname])
        return logging.Formatter.format(self, record)



class IRCColoredFormatter(logging.Formatter):
    WHITE, BLACK, BLUE, GREEN, RED, BROWN, PURPLE, ORANGE, YELLOW, LIGHT_GREEN, \
            TEAL, LIGHT_CYAN, LIGHT_BLUE, MAGENTA, GREY, LIGHT_GREY = range(16)

    RESET_SEQ = "\x03"
    COLOR_SEQ = "\x03%d"

    COLORS = {
        'WARNING': ORANGE,
        'INFO': BLACK,
        'DEBUG': BLUE,
        'CRITICAL': MAGENTA,
        'ERROR': RED
    }

    def format(self, record):  # @ReservedAssignment
        levelname = record.levelname
        if levelname in self.COLORS:
            return self.COLOR_SEQ % self.COLORS[levelname] + logging.Formatter.format(self, record)
        else:
            return logging.Formatter.format(self, record)


class IRCLogHandler(logging.Handler):
    """ Dumps usual log output to some IRC channel
    """

    def __init__(self, client, channel, level=logging.NOTSET):
        logging.Handler.__init__(self, level=level)
        self.client = client
        self.channel = channel
        self.client.join_channel(self.channel)

    def emit(self, record):
        self.client.msg(self.channel, self.format(record))


class SubjectSMTPHandler(SMTPHandler):
    """ Send log messages as emails with meaningful subject 
    """
    def __init__(self, prefix, *args, **kwargs):
        kwargs['secure'] = True
        SMTPHandler.__init__(self, *args, **kwargs)
        self.prefix = prefix

    def getSubject(self, record):
        text = record.message.split('\n')[0]
        if len(text) > 40: text = text[:35] + '...'
        return "%s: %s: %s" % (self.prefix, record.levelname, text)

    def handleError(self, record):
        logger.exception("SMTP logging failed")
        logger.handle(record)


class SMSHandler(HTTPHandler):
    """ Send log messages as SMS via smstrade.de 
    """
    valid_sender_chars = frozenset(string.ascii_letters + string.digits)

    def __init__(self, sender, receiver, key, errors='log'):
        HTTPHandler.__init__(self, "gw.cmtelecom.com", "https://gw.cmtelecom.com/v1.0/message", method="POST")
        self.key = key
        self.sender = self.fix_sender(sender)
        self.receiver = receiver.replace('+', '00')
        self.error_behaviour = errors

    def fix_sender(self, sender):
        sender = sender.strip()
        if sender.startswith('+'):
            sender = sender.replace('+', '00')
        sender = ''.join(c for c in sender if c in self.valid_sender_chars)
        if any(c not in string.digits for c in sender):
            sender = sender[:11]
        return sender

    def mapLogRecord(self, record, reference='Log'):
        intro_str = record.levelname + ' '
        message = intro_str + record.getMessage()[:160 - len(intro_str)]

        msg = {'from': self.sender,
               'to': [{'number': self.receiver}],
               'body': {'content': message}}
        if reference:
            msg['reference'] = reference
        return {'messages': {'authentication': {'producttoken': self.key},
                             'msg': [msg]}}

    def emit(self, record):
        """
        Emit a record.

        Send the record to the Web server as a percent-encoded dictionary
        """
        try:
            import requests
            data = self.mapLogRecord(record)
            resp = requests.post(self.url, json=data)

            if resp.status_code != 200:
                raise RuntimeError("Unsuccessful SMS provider response (%s): %s" % (resp.status_code, resp.json()['details']))
            for msg in resp.json()['messages']:
                if msg['status'] != 'Accepted':
                    raise RuntimeError("Could not deliver message: %s" % msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

    def handleError(self, record):
        if self.error_behaviour == 'log':
            logger.exception("SMS logging failed")
            logger.handle(record)
        else:
            raise


class BufferingHandler(StreamHandler):
    def __init__(self):
        self.buffer = StringIO.StringIO()
        StreamHandler.__init__(self, self.buffer)

    def emit(self, record):
#        self.flush()
        StreamHandler.emit(self, record)

    @property
    def content(self):
        self.buffer.seek(0)
        return self.buffer.read()

    @property
    def lines(self):
        return self.content.split('\n')

    def flush(self):
        self.buffer.flush()


class AdvancedTimedRotatingFileHandler(TimedRotatingFileHandler):
    """ Ergaenzung von TimedRotatingFileHandler um einen Check beim Initialisieren,
        um Logfiles auch dann zu rotieren, wenn die aktuelle Datei veraltet ist.
        D.h. falls die Logdatei laenger als das Rotationsintervall nicht mehr geaendert
        wurde, wird ebenfalls rotiert.
    """
    def __init__(self, filename, **kwargs):
        touched = os.stat(filename).st_mtime
        TimedRotatingFileHandler.__init__(self, filename, **kwargs)
        if self.rolloverAt - self.interval > touched:
            self.doRollover()
