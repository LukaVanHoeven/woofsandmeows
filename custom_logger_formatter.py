"""
DISCLAIMER: 
This code was previously part of Joris Heemskerks Bachelors thesis, 
and is being re-used here. All rights are reserved to Joris Heemskerk, 
and Technolution BV, Gouda NL. Joris was granted the rights to use and 
modify this code, at the express notion that a disclaimer was put in.
"""
""" 
Code is based on:
https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
"""

import datetime
import logging
import pytz


class CustomLoggerFormatter(logging.Formatter):
    """ 
    Custom formatter for logging

    This formatter gives colors to the different styles of logging, 
    see `self.FORMATS`.
    """
    grey = "\x1b[38;20m"
    dark_grey = "\x1b[30m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: dark_grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def converter(self, timestamp):
        """ 
        Convert the timestamp to Europe/Amsterdam timezone.

        :param timestamp: Timestamp for logging message.
        :type  timestamp: Timestamp (same as time.time()).
        """
        dt = datetime.datetime.fromtimestamp(
            timestamp, pytz.timezone('Europe/Amsterdam')
        )
        return dt.timetuple()

    def format(self, record: logging.LogRecord):
        """
        Formatter method.
        
        This method formats the logging messages using `self.FORMATS`.

        :param record: Record to log
        :type  record: LogRecord
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatter.converter = self.converter
        return formatter.format(record)
