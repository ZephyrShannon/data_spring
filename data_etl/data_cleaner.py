import pandas as pd
from datetime import datetime


class DataCleaner:
    def __init__(self, start_dt: datetime, end_dt: datetime, freq):
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.freq = freq



    def clean(self, data: pd.DataFrame):
        return data

