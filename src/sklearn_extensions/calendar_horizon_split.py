from datetime import date, datetime
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from sklearn.model_selection import BaseCrossValidator


class CalendarHorizonSplit(BaseCrossValidator):
    def __init__(
        self,
        date_column: str,
        horizon_length_days: int,
        validation_length_days: int,
        n_splits: int = 5,
    ) -> None:
        self.date_column = date_column
        self.horizon_length = pd.Timedelta(days=horizon_length_days)
        self.validation_length = pd.Timedelta(days=validation_length_days)
        self.n_splits = n_splits

    def split(
        self, 
        X: pd.DataFrame,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: # np.intp is just a signed integer used for indexing arrays
        if self.date_column not in X.columns:
            raise KeyError(f"The date column '{self.date_column}' does not exist.")
        
        dates = pd.to_datetime(X[self.date_column])
        if dates.isna().any(): # If ANY of the date values is missing, then raise an error
            raise ValueError(f"The date column '{self.date_column}' contains {X[self.date_column].isna().sum()} missing values.")

        # Get the date of the last observation
        end_date = dates.max().normalize() + pd.Timedelta(days=1) # Normalize sets the time to 00:00:00

        for i in range(self.n_splits):
            validation_start_date = end_date - self.validation_length * (self.n_splits - i) # Work backwards from the end date
            validation_end_date = validation_start_date + self.validation_length

            train_mask = dates + self.horizon_length < validation_start_date
            validation_mask = (dates >= validation_start_date) & (dates < validation_end_date)

            train_idx = np.flatnonzero(train_mask) # Get an array of observation indexes from X subject to the training mask
            validation_idx = np.flatnonzero(validation_mask) # Get an array of observation indexes from X subject to the validation mask

            if len(train_idx) == 0 or len(validation_idx) == 0:
                raise ValueError(f"Empty fold {i + 1}")

            yield train_idx, validation_idx

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        return self.n_splits
                     