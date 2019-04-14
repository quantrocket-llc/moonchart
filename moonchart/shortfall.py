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
import matplotlib.pyplot as plt
from .base import BaseTearsheet
from .perf import DailyPerformance
from .exceptions import MoonchartError
from .utils import with_baseline, get_cum_returns, get_drawdowns

class ShortfallTearsheet(BaseTearsheet):

    @classmethod
    def from_csvs(cls, sim_filepath_or_buffer, live_filepath_or_buffer,
                  start_date=None,
                  columns=None,
                  ignore_top_n=None,
                  figsize=None, trim_outliers=None,
                  pdf_filename=None, riskfree=0,
                  compound=True, rolling_sharpe_window=200):
        """
        Create a shortfall tear sheet from a Moonshot backtest results CSV and PNL csv.

        Parameters
        ----------
        sim_filepath_or_buffer : str or file-like object
            filepath or file-like object of the Moonshot backtest CSV

        live_filepath_or_buffer : str or file-like object
            filepath or file-like object of the PNL CSV

        start_date : str (YYYY-MM-DD), optional
            truncate at start date (otherwise include entire date range)

        columns : list of str, optional
            limit to showing shortfall for these columns (default is all intersecting columns)

        ignore_top_n : int, optional
            ignore this many of the largest magnitude discrepancies between live and
            simulated. Default is None, meaning don't ignore anything.

        figsize : tuple (width, height), optional
            (width, height) of matplotlib figure. Default is (16, 12)

        trim_outliers: int or float, optional
            discard returns that are more than this many standard deviations
            from the mean

        pdf_filename : string, optional
            save tear sheet to this filepath as a PDF instead of displaying

        riskfree : float, optional
            the riskfree rate (default 0)

        compound : bool
            True for compound/geometric returns, False for arithmetic returns.
            Default True

        rolling_sharpe_window : int, optional
            compute rolling Sharpe over this many periods (default 200)

        Returns
        -------
        None

        Examples
        --------
        >>> from moonshot import ShortfallTearsheet
        >>> ShortfallTearsheet.from_csvs("backtest.csv", "pnl.csv")
        """

        sim_perf = DailyPerformance.from_moonshot_csv(
            sim_filepath_or_buffer,
            trim_outliers=trim_outliers,
            riskfree=riskfree,
            compound=compound,
            rolling_sharpe_window=rolling_sharpe_window)

        live_results = pd.read_csv(live_filepath_or_buffer,
                                   parse_dates=["Date"])
        live_results.loc[:, "Date"] = live_results.Date.dt.normalize()
        live_results = live_results.set_index(["Field","Date"])


        live_perf = DailyPerformance._from_pnl(
            live_results,
            trim_outliers=trim_outliers,
            riskfree=riskfree,
            compound=compound,
            rolling_sharpe_window=rolling_sharpe_window)

        t = cls(figsize=figsize, pdf_filename=pdf_filename)
        return t.create_shortfall_tearsheet(
            sim_perf, live_perf,
            start_date=start_date,
            columns=columns,
            ignore_top_n=ignore_top_n
        )

    def create_shortfall_tearsheet(self, sim_perf, live_perf, start_date=None,
                                   columns=None, ignore_top_n=None):
        """
        Shows drawdown and equity curves comparing the live and simulated
        returns. DataFrames should include a column for each strategy being
        analyzed.
        """

        live_returns = live_perf.returns
        sim_returns = sim_perf.returns

        if start_date:
            live_returns = live_returns[live_returns.index >= start_date]
            sim_returns = sim_returns[sim_returns.index >= start_date]

        strategies = live_returns.columns.intersection(sim_returns.columns)
        if columns:
            strategies = set(strategies).intersection(set(columns))

        strategy_count = len(strategies)

        if not strategy_count:
            raise MoonchartError("simulated and live CSVs contain no overlapping columns")

        for strategy in strategies:
            strategy_live_returns = live_returns[strategy]
            strategy_live_returns.name = "live"
            strategy_simulated_returns = sim_returns[strategy]
            strategy_simulated_returns.name = "simulated"
            try:
                strategy_returns = pd.concat((strategy_live_returns, strategy_simulated_returns), axis=1, sort=True).fillna(0)
            except TypeError:
                # sort was introduced in pandas 0.23
                strategy_returns = pd.concat((strategy_live_returns, strategy_simulated_returns), axis=1).fillna(0)

            if ignore_top_n:
                shortfall = strategy_returns.live - strategy_returns.simulated
                top_n_shortfall_cutoff = shortfall.abs().sort_values(ascending=False).head(ignore_top_n).iloc[-1]
                shortfall = strategy_returns.apply(lambda x: shortfall)
                ignore = shortfall.abs() >= top_n_shortfall_cutoff
                strategy_returns = strategy_returns.where(~ignore)

            cum_returns = get_cum_returns(strategy_returns)
            cum_shortfall = cum_returns.live - cum_returns.simulated

            if ignore_top_n:
                cum_shortfall = cum_shortfall.fillna(method="ffill")

            fig = plt.figure(strategy + " Returns", figsize=self.figsize)
            axis = fig.add_subplot(311)
            with_baseline(cum_returns).plot(ax=axis, title=strategy + " Cumulative Returns")
            self._y_format_at_least_two_decimal_places(axis)
            axis.set_ylabel("Cumulative return")
            axis.set_xlabel("")

            fig = plt.figure(strategy + " Shortfall", figsize=self.figsize)
            axis = fig.add_subplot(312)
            axis.set_ylabel("Shortfall")
            self._y_format_as_percentage(axis)
            with_baseline(cum_shortfall, value=0).plot(ax=axis, title=strategy + " Cumulative Shortfall", kind="area", stacked=False)
            axis.set_xlabel("")

            axis = fig.add_subplot(313)
            axis.set_ylabel("Return")
            self._y_format_as_percentage(axis)
            shortfall = strategy_returns.live - strategy_returns.simulated
            strategy_returns["shortfall"] = shortfall
            strategy_returns = with_baseline(strategy_returns, value=0)
            strategy_returns.index = strategy_returns.index.strftime("%Y-%m-%d")
            strategy_returns.plot(ax=axis, title=strategy + " Returns", kind="bar")
            axis.set_xlabel("")

            if sim_perf.commissions is not None and live_perf.commissions is not None:
                strategy_live_commissions = live_perf.commissions[strategy]
                strategy_live_commissions.name = "live"
                strategy_simulated_commissions = sim_perf.commissions[strategy]
                strategy_simulated_commissions.name = "simulated"
                try:
                    strategy_commissions = pd.concat((strategy_live_commissions, strategy_simulated_commissions), axis=1, sort=True).fillna(0)
                except TypeError:
                    # sort was introduced in pandas 0.23
                    strategy_commissions = pd.concat((strategy_live_commissions, strategy_simulated_commissions), axis=1).fillna(0)

                if start_date:
                    strategy_commissions = strategy_commissions[strategy_commissions.index >= start_date]

                fig = plt.figure(strategy + " Commissions", figsize=self.figsize)
                cum_commissions = get_cum_returns(strategy_commissions)

                axis = fig.add_subplot(211)
                with_baseline(cum_commissions).plot(ax=axis, title=strategy + " Cumulative Commissions")
                self._y_format_at_least_two_decimal_places(axis)
                axis.set_ylabel("Commissions")
                axis.set_xlabel("")

                axis = fig.add_subplot(212)
                strategy_commissions.plot(ax=axis, kind="bar", title=strategy + " Commissions")
                self._y_format_as_percentage(axis)
                axis.set_ylabel("Commissions")
                axis.set_xlabel("")

        self._save_or_show()
