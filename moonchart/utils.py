# Copyright 2017-2023 QuantRocket LLC - All Rights Reserved
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
"""
Utility functions for performance analysis.

Functions
---------
get_sharpe
    Return the Sharpe ratio of the returns.

get_rolling_sharpe
    Compute rolling Sharpe ratios for the returns.

get_cum_returns
    Compute the cumulative returns of the provided returns.

get_zscores
    Return the Z-scores of the input returns.

get_cagr
    Compute the CAGR from the cumulative returns.

get_drawdowns
    Compute the drawdowns of the cumulative returns.

get_top_movers
    Return the biggest gainers and losers in the returns.

intraday_to_daily
    Roll up a DataFrame of intraday performance results to daily, dropping
    the "Time" level from the multi-index.

trim_outliers
    Zero out observations that are too many standard deviations from the
    mean.

with_baseline
    Prepend a date-indexed Series or DataFrame with an initial row that is
    one period earlier than the first row and has the specified value.
"""
from typing import overload
import seaborn as sns
import pandas as pd
import numpy as np
from .exceptions import MoonchartError

__all__ = [
    "get_sharpe",
    "get_rolling_sharpe",
    "get_cum_returns",
    "get_zscores",
    "get_cagr",
    "get_drawdowns",
    "get_top_movers",
    "intraday_to_daily",
    "trim_outliers",
    "with_baseline",
]

def set_default_palette():
    """
    Sets the default palette so that the first 3 colors are
    blue, green, red. This was the case in Matplotlib 2 but in
    Matplotlib the default sequence is blue, orange, green, red,
    which is not as good for Moonchart plots. This function
    will remove orange from position 2 and put it at the end.
    """
    # Set seaborn default style
    sns.set()

    current_palette = sns.color_palette()
    orange = (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
    if current_palette[1] == orange:
        current_palette.append(current_palette.pop(1))
        sns.set_palette(current_palette)

@overload
def get_zscores(
    returns: 'pd.Series[float]'
    ) -> 'pd.Series[float]':
    pass

@overload
def get_zscores(
    returns: pd.DataFrame
    ) -> pd.DataFrame:
    pass

def get_zscores(returns):
    """
    Return the Z-scores of the input returns.

    Parameters
    ----------
    returns : Series or DataFrame, required
        Series or DataFrame of returns

    Returns
    -------
    Series or DataFrame
    """
    # Ignore 0 returns in calculating z score
    nonzero_returns = returns.where(returns != 0)
    z_scores = (nonzero_returns - nonzero_returns.mean())/nonzero_returns.std()
    return z_scores

@overload
def trim_outliers(
    returns: 'pd.Series[float]',
    z_score: float
    ) -> 'pd.Series[float]':
    pass

@overload
def trim_outliers(
    returns: pd.DataFrame,
    z_score: float
    ) -> pd.DataFrame:
    pass

def trim_outliers(returns, z_score):
    """
    Zero out observations that are too many standard deviations from the
    mean.

    Parameters
    ----------
    returns : Series or DataFrame, required
        Series or DataFrame of returns

    z_score : int or float, required
        maximum standard deviation values are allowed to be from the mean

    Returns
    -------
    Series or DataFrame
    """
    z_scores = get_zscores(returns)
    return returns.where(z_scores.abs() <= z_score, 0)

def with_baseline(data, value=1):
    """
    Prepend a date-indexed Series or DataFrame with an initial row that is
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

def _pad_returns(returns):
    """
    Pads a returns Series or DataFrame with business days, in case the
    existing Date index is sparse (as with PNL csvs). Sparse indexes if not
    padded will affect the Sharpe ratio because the 0 return days will not be
    included in the mean and std.
    """
    bdays = pd.date_range(start=returns.index.min(), end=returns.index.max(),freq="B")
    idx = returns.index.union(bdays)
    return returns.reindex(index=idx).fillna(0)

def _get_sharpe(returns, riskfree=0):
    """
    Private function that returns the Sharpe ratio of the returns. Returns
    should already be padded when calling this function.
    """
    mean = (returns - riskfree).mean()
    if isinstance(mean, float) and mean == 0:
        return 0
    std = (returns - riskfree).std()
    # Returns are assumed to represent daily returns, so annualize the Sharpe ratio
    return mean/std * np.sqrt(252)

@overload
def get_sharpe(
    returns: 'pd.Series[float]',
    riskfree: float = 0
    ) -> float:
    pass

@overload
def get_sharpe(
    returns: pd.DataFrame,
    riskfree: float = 0
    ) -> 'pd.Series[float]':
    pass

def get_sharpe(returns, riskfree=0):
    """
    Return the Sharpe ratio of the returns.

    Parameters
    ----------
    returns : Series or DataFrame, required
        a Series or DataFrame of returns

    riskfree : float, optional
        the risk-free rate (default 0)

    Returns
    -------
    float or Series of floats
    """
    returns = _pad_returns(returns)
    return _get_sharpe(returns, riskfree=riskfree)

@overload
def get_rolling_sharpe(
    returns: 'pd.Series[float]',
    window: int,
    riskfree: float = 0
    ) -> 'pd.Series[float]':
    pass

@overload
def get_rolling_sharpe(
    returns: pd.DataFrame,
    window: int,
    riskfree: float = 0
    ) -> pd.DataFrame:
    pass

def get_rolling_sharpe(returns, window, riskfree=0):
    """
    Compute rolling Sharpe ratios for the returns.

    Parameters
    ----------
    returns : Series or DataFrame, required
        a Series or DataFrame of returns

    window : int, required
        rolling window length

    riskfree : float, optional
        the risk-free rate (default 0)

    Returns
    -------
    Series or DataFrame
    """
    returns = _pad_returns(returns)
    rolling_returns = returns.rolling(window, min_periods=window)
    try:
        return rolling_returns.apply(_get_sharpe, raw=True, kwargs=dict(riskfree=riskfree))
    except TypeError as e:
        # handle pandas<0.23
        if "apply() got an unexpected keyword argument 'raw'" in repr(e):
            return rolling_returns.apply(_get_sharpe, kwargs=dict(riskfree=riskfree))
        else:
            raise

@overload
def get_cum_returns(
    returns: 'pd.Series[float]',
    compound: bool = True
    ) -> 'pd.Series[float]':
    pass

@overload
def get_cum_returns(
    returns: pd.DataFrame,
    compound: bool = True
    ) -> pd.DataFrame:
    pass

def get_cum_returns(returns, compound=True):
    """
    Compute the cumulative returns of the provided returns.

    Parameters
    ----------
    returns : Series or DataFrame, required
        a Series or DataFrame of returns

    compound : bool
        True for compounded (geometric) returns, False for arithmetic
        returns (default True)

    Returns
    -------
    Series or DataFrame
    """
    if compound:
        cum_returns = (1 + returns).cumprod()
    else:
        cum_returns = returns.cumsum() + 1

    cum_returns.index.name = "Date"
    return cum_returns

@overload
def get_cagr(
    cum_returns: 'pd.Series[float]',
    compound: bool = True
    ) -> float:
    pass

@overload
def get_cagr(
    cum_returns: pd.DataFrame,
    compound: bool = True
    ) -> 'pd.Series[float]':
    pass

def get_cagr(cum_returns, compound=True):
    """
    Compute the CAGR from the cumulative returns.

    Parameters
    ----------
    cum_returns : Series or DataFrame, required
        a Series or DataFrame of cumulative returns

    compound : bool
        compute compound annual growth rate if True, otherwise
        compute average annual return (default True)

    Returns
    -------
    float or Series of floats
    """
    # For DataFrames, apply this function to each Series.
    if isinstance(cum_returns, pd.DataFrame):
        return cum_returns.apply(get_cagr, axis=0)

    # Ignore nulls when compting CAGR
    cum_returns = cum_returns[cum_returns.notnull()]

    if cum_returns.empty:
        return 0

    # Compute the CAGR of the Series
    min_date = cum_returns.index.min()
    max_date = cum_returns.index.max()
    years = ((max_date - min_date).days or 1)/365.0
    ending_value = cum_returns.iloc[-1]
    # Since we are computing CAGR on cumulative returns, the beginning
    # value is always 1.
    beginning_value = 1
    if compound:
        cagr = (ending_value/beginning_value)**(1/years) - 1
    else:
        # Compound annual growth rate doesn't apply to arithmetic
        # returns, so just divide the cum_returns by the number of years
        # to get the annual return
        cagr = (ending_value/beginning_value - 1)/years

    return cagr

@overload
def get_drawdowns(
    cum_returns: 'pd.Series[float]'
    ) -> 'pd.Series[float]':
    pass

@overload
def get_drawdowns(
    cum_returns: pd.DataFrame
    ) -> pd.DataFrame:
    pass

def get_drawdowns(cum_returns):
    """
    Compute the drawdowns of the cumulative returns.

    Parameters
    ----------
    cum_returns : Series or DataFrame, required
        a Series or DataFrame of cumulative returns

    Returns
    -------
    Series or DataFrame
    """
    cum_returns = cum_returns[cum_returns.notnull()]
    highwater_marks = cum_returns.expanding().max()
    drawdowns = cum_returns/highwater_marks - 1
    return drawdowns

@overload
def get_top_movers(
    returns: 'pd.Series[float]',
    n: int = 10
    ) -> 'pd.Series[float]':
    pass

@overload
def get_top_movers(
    returns: pd.DataFrame,
    n: int = 10
    ) -> pd.DataFrame:
    pass

def get_top_movers(returns, n=10):
    """
    Return the biggest gainers and losers in the returns.

    Parameters
    ----------
    returns : Series or DataFrame, required
        a Series or DataFrame of returns

    n : int, optional
        the number of biggest gainers and losers to return (default 10)

    Returns
    -------
    Series or DataFrame
    """

    if isinstance(returns, pd.DataFrame):
        returns = returns.stack()

    returns = returns.sort_values()

    try:
        top_movers = pd.concat((returns.head(n), returns.tail(n)), sort=True)
    except TypeError:
        # sort was introduced in pandas 0.23
        top_movers = pd.concat((returns.head(n), returns.tail(n)))

    return top_movers

def intraday_to_daily(
    results: pd.DataFrame,
    how: dict[str, str] = None
    ) -> pd.DataFrame:
    """
    Roll up a DataFrame of intraday performance results to daily, dropping
    the "Time" level from the multi-index.

    The following aggregation methods are supported:

    extreme: min or max of day, whichever is of greater absolute magnitude
    last: last value of day
    max: max of day
    mean: mean of day
    sum: sum of day

    By default, supported fields are aggregated as follows:

    AbsExposure: max
    AbsWeight: max
    Benchmark: sum
    Commission: sum
    CommissionAmount: sum
    NetExposure: extreme
    NetLiquidation: mean
    Pnl: sum
    PositionQuantity: extreme
    PositionValue: extreme
    Price: last
    Return: sum
    Slippage: sum
    TotalHoldings: max
    Turnover: sum
    Weight: extreme

    This can be overridden with the `how` parameter.

    Parameters
    ----------
    results : DataFrame, required
         a DataFrame of intraday Moonshot backtest results or PNL results, with
         a "Time" level in the index

    how : dict, optional
        a dict of {fieldname: aggregation method} specifying how to aggregate
        fields. This is combined with and overrides the default methods.

    Returns
    -------
    DataFrame
        a DataFrame of daily results, without a "Time" level in the index

    Examples
    --------
    Convert intraday Moonshot results to daily:

    >>> intraday_results = read_moonshot_csv("moonshot_intraday_backtest.csv")
    >>> daily_results = intraday_to_daily(intraday_results)
    """
    if "Time" not in results.index.names:
        raise MoonchartError("results DataFrame must have 'Time' level in index")

    fields_in_results = results.index.get_level_values("Field").unique()

    daily_results = {}

    # how to aggregate by field
    field_hows = {
        'AbsExposure': 'max',
        'AbsWeight': 'max',
        'Benchmark': 'sum',
        'Commission': 'sum',
        'CommissionAmount': 'sum',
        'NetExposure': 'extreme',
        'NetLiquidation': 'mean',
        'Pnl': 'sum',
        'PositionQuantity': 'extreme',
        'PositionValue': 'extreme',
        'Price': 'last' ,
        'Return': 'sum',
        'Slippage': 'sum',
        'TotalHoldings': 'max',
        'Turnover': 'sum',
        'Weight': 'extreme',
    }

    if how:
        field_hows.update(how)

    for field in fields_in_results:
        if field not in field_hows:
            continue

        field_how = field_hows[field]

        field_results = results.loc[field].astype(np.float64)
        grouped = field_results.groupby(field_results.index.get_level_values("Date"))

        if field_how == "extreme":
            mins = field_results.groupby(field_results.index.get_level_values("Date")).min()
            maxes = field_results.groupby(field_results.index.get_level_values("Date")).max()
            daily_results[field] = mins.where(mins.abs()>maxes.abs(), maxes)
        else:
            daily_results[field] = getattr(grouped, field_how)()

    daily_results = pd.concat(daily_results, names=["Field","Date"])
    return daily_results