"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's prior
work for the Computer Vision course, and is being re-used here.
"""

import datetime
import pytz


date = datetime.datetime.now(
    tz=pytz.timezone('Europe/Amsterdam')
).strftime('%d-%m-%Y--%H-%M')
OUTPUT_DIR = f"output/{date}/"
