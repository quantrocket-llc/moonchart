# Copyright 2019 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

def trim_outliers(data, z_score):
    """
    Zeroes out observations that are too many standard deviations from the
    mean.

    Parameters
    ----------
    data : Series or DataFrame, required
        Series or DataFrame of observations

    z_score : int or float, required
        maximum standard deviation values are allowed to be from the mean

    Returns
    -------
    Series or DataFrame
    """
    z_scores = (data - data.mean())/data.std()
    return data.where(z_scores.abs() <= z_score, 0)

def with_baseline(data, value=1):
    """
    Prepends a date-indexed Series or DataFrame with an initial row that is
    one period earlier than the first row and has the specified value.

    The typical use case is for generating plots: without a baseline row, a cumulative
    returns plot won't start from 1 if the first day's return is nonzero.

    Parameters
    ----------
    data : Series or DataFrame, required
        Series or DataFrame (for example, of returns)

    value : required
        value to insert in the baseline row

    Returns
    -------
    Series or DataFrame

    Examples
    --------
    Typical usage:

    >>> with_baseline(cum_returns).plot()

    Under the hood:

    >>> cum_returns.head()
        2019-01-02   1.01
        2019-01-03  0.995
    >>> with_baseline(cum_returns)
        2019-01-01      1
        2019-01-02   1.01
        2019-01-03  0.995
    """
    period_length = data.index[1] - data.index[0]
    prior_period = data.index[0] - period_length
    if isinstance(data, pd.DataFrame):
        baseline_row = pd.DataFrame(value, index=[prior_period], columns=data.columns)
    else:
        baseline_row = pd.Series(value, index=[prior_period], name=data.name)
    try:
        data_with_baseline = pd.concat((baseline_row, data), sort=False)
    except TypeError:
        # sort was introduced in pandas 0.23
        data_with_baseline = pd.concat((baseline_row, data))
    return data_with_baseline
