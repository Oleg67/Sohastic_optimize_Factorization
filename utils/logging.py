from __future__ import absolute_import
import logging


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
