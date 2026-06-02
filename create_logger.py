"""
DISCLAIMER: 
This code was previously part of Joris Heemskerks Bachelors thesis, 
and is being re-used here. All rights are reserved to Joris Heemskerk, 
and Technolution BV, Gouda NL. Joris was granted the rights to use and 
modify this code, at the express notion that a disclaimer was put in.
"""
import logging
import sys

from logging.handlers import RotatingFileHandler

from custom_logger_formatter import CustomLoggerFormatter


def create_logger(
    name: str, 
    output_log_file_name: str | None=None, 
    level: int=logging.DEBUG
)-> logging.Logger:
    """ 
    Create logger with custom formatter

    Also logs to file if provided.

    :param name: Name of the specific logger.
    :type  name: str
    :param output_log_file_name: File to log output to 
        (please end this file in .log). (DEFAULT=None)
    :type  output_log_file_name: str | None
    :param level: Logging level. (DEFAULT=logging.DEBUG)
    :type  level: int

    :return: Set up logger that has already logged a starting message.
    :rtype: logging.Logger 
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  
    ch = _make_stream_handler(level)
    logger.addHandler(ch)
    if output_log_file_name is not None:
        f_ch = RotatingFileHandler(output_log_file_name)
        f_ch.setLevel(level)
        f_ch.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(f_ch)

    return logger

def create_basic_logger(name, level: int=logging.DEBUG)-> logging.Logger:
    """ 
    Create simple logger with custom formatter
    
    :param name: Name of the specific logger.
    :type  name: str
    :param level: Logging level. (DEFAULT=logging.DEBUG)
    :type  level: int

    :rtype: logging.Logger 
    :return: Set up logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  
    return logger

def _make_stream_handler(level: int) -> logging.StreamHandler:
    """
    Create a UTF-8 stream handler to avoid encoding errors on Windows.

    :param level: Logging level.
    :type level: int
    :returns: stream handler.
    :rtype: logging.StreamHandler
    """
    stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
    ch = logging.StreamHandler(stream)
    ch.setLevel(level)
    ch.setFormatter(CustomLoggerFormatter())
    return ch
