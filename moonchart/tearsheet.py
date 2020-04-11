# Copyright 2017 QuantRocket LLC - All Rights Reserved
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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math
import warnings
import empyrical as ep
import scipy.stats
from .perf import DailyPerformance, AggregateDailyPerformance
from .base import BaseTearsheet
from .exceptions import MoonchartError
from .utils import (
    with_baseline,
    get_sharpe,
    get_cagr,
    get_cum_returns,
    get_drawdowns
)

class Tearsheet(BaseTearsheet):
    """
    Generates a tear sheet of performance stats and graphs.
    """
    @classmethod
    def from_moonshot_csv(cls, filepath_or_buffer, figsize=None,
                          max_cols_for_details=25, trim_outliers=None,
                          how_to_aggregate=None,
                          pdf_filename=None, riskfree=0,
                          compound=True, rolling_sharpe_window=200):
        """
        Create a full tear sheet from a moonshot backtest results CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath of CSV or file-like object

        figsize : tuple (width, height), optional
            (width, height) of matplotlib figure. Default is (16, 12)

        max_cols_for_details : int, optional
            suppress detailed plots if there are more than this many columns
            (i.e. strategies or securities). Too many plots may cause slow
            rendering. Default 25.

        trim_outliers: int or float, optional
            discard returns that are more than this many standard deviations
            from the mean. Useful for dealing with data anomalies that cause
            large spikes in plots.

        how_to_aggregate : dict, optional
            a dict of {fieldname: aggregation method} specifying how to aggregate
            fields from intraday to daily. See the docstring for
            `moonchart.utils.intraday_to_daily` for more details.

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
        >>> from moonshot import Tearsheet
        >>> Tearsheet.from_moonshot_csv("backtest_results.csv")
        """
        perf = DailyPerformance.from_moonshot_csv(
            filepath_or_buffer,
            trim_outliers=trim_outliers,
            how_to_aggregate=how_to_aggregate,
            riskfree=riskfree,
            compound=compound,
            rolling_sharpe_window=rolling_sharpe_window)

        t = cls(figsize=figsize,
                max_cols_for_details=max_cols_for_details,
                pdf_filename=pdf_filename)

        return t.create_full_tearsheet(perf)

    @classmethod
    def from_pnl_csv(cls, filepath_or_buffer, figsize=None,
                     max_cols_for_details=25, trim_outliers=None,
                     how_to_aggregate=None,
                     pdf_filename=None, riskfree=0,
                     compound=True, rolling_sharpe_window=200):
        """
        Create a full tear sheet from a pnl CSV.

        Parameters
        ----------
        filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV

        figsize : tuple (width, height), optional
            (width, height) of matplotlib figure. Default is (16, 12)

        max_cols_for_details : int, optional
            suppress detailed plots if there are more than this many columns
            (i.e. strategies or securities). Too many plots may cause slow
            rendering. Default 25.

        trim_outliers: int or float, optional
            discard returns that are more than this many standard deviations
            from the mean

        how_to_aggregate : dict, optional
            a dict of {fieldname: aggregation method} specifying how to aggregate
            fields from intraday to daily. See the docstring for
            `moonchart.utils.intraday_to_daily` for more details.

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
        """
        perf = DailyPerformance.from_pnl_csv(
            filepath_or_buffer,
            trim_outliers=trim_outliers,
            how_to_aggregate=how_to_aggregate,
            riskfree=riskfree,
            compound=compound,
            rolling_sharpe_window=rolling_sharpe_window)

        t = cls(figsize=figsize,
                max_cols_for_details=max_cols_for_details,
                pdf_filename=pdf_filename)

        return t.create_full_tearsheet(perf)

    def create_full_tearsheet(self, performance):
        """
        Create a full tear sheet of performance results including returns
        plots, returns by year plots, and position-related plots.

        Parameters
        ----------
        performance : instance
            a DailyPerformance instance

        Returns
        -------
        None

        Examples
        --------
        >>> from moonchart import DailyPerformance, Tearsheet
        >>> perf = DailyPerformance.from_moonshot_csv("backtest_results.csv")
        >>> t = Tearsheet()
        >>> t.create_full_tearsheet(perf)

        See Also
        --------
        Tearsheet.from_moonshot_csv : create a full tear sheet from a Moonshot CSV
        """
        agg_performance = AggregateDailyPerformance(performance)

        num_cols = len(performance.returns.columns)
        if num_cols > self.max_cols_for_details:
            warnings.warn("Suppressing details because there are more than {0} columns "
                          "(you can control this setting by modifying "
                          "Tearsheet.max_cols_for_details)".format(
                              self.max_cols_for_details))

        self.create_summary_tearsheet(performance, agg_performance)
        self.create_returns_tearsheet(performance, agg_performance)
        self.create_returns_by_year_tearsheet(performance, agg_performance)

        if any([exposures is not None for exposures in (
            performance.net_exposures, performance.abs_exposures)]):
            self.create_positions_tearsheet(performance, agg_performance)

        self._create_constituents_tearsheet(performance)

        self._save_or_show()

    def create_summary_tearsheet(self, performance, agg_performance=None):
        """
        Create a tear sheet of summary performance stats in a table.

        Parameters
        ----------
        performance : DailyPerformance, required
            a DailyPerformance instance

        agg_performance : AggregateDailyPerformance, optional
            an AggregateDailyPerformance instance. Constructed from performance
            if not provided.

        Returns
        -------
        None

        Examples
        --------
        >>> from moonchart import DailyPerformance, Tearsheet
        >>> perf = DailyPerformance.from_moonshot_csv("backtest_results.csv")
        >>> t = Tearsheet()
        >>> t.create_summary_tearsheet(perf)
        """
        if agg_performance is None:
            agg_performance = AggregateDailyPerformance(performance)

        stats = []

        if agg_performance.pnl is not None:
            stats.append(["PNL", round(agg_performance.pnl.sum(), 2)])
        if agg_performance.commission_amounts is not None:
            stats.append(["Commissions", round(agg_performance.commission_amounts.sum(), 2)])

        stats.append(["Start Date", agg_performance.returns.index.min().date().isoformat()])
        stats.append(["End Date", agg_performance.returns.index.max().date().isoformat()])

        stats.append(['Total Months', round(
            (agg_performance.returns.index.max() - agg_performance.returns.index.min()) / pd.Timedelta(365.25/12, 'D'))])

        stats.append(["", " Risk and Returns"])
        stats.append(["CAGR", "{0}%".format(round(agg_performance.cagr * 100, 1))])
        stats.append([
            "Sharpe Ratio",
            '%.2f' % agg_performance.sharpe])
        stats.append([
            "Max Drawdown",
            "{0}%".format(round(agg_performance.max_drawdown * 100, 1))])
        stats.append([
            "Cumulative Return",
            "{0}%".format(round(ep.cum_returns_final(agg_performance.returns) * 100, 1))])
        stats.append([
            "Annual Volatility",
            "{0}%".format(round(ep.annual_volatility(agg_performance.returns) * 100, 1))])
        stats.append([
            "Sortino Ratio",
            '%.2f' % ep.sortino_ratio(agg_performance.returns)])
        stats.append([
            "Calmar Ratio",
            '%.2f' % ep.calmar_ratio(agg_performance.returns)])
        stats.append([
            "Skew",
            '%.2f' % scipy.stats.skew(agg_performance.returns)])
        stats.append([
            "Kurtosis",
            '%.2f' % scipy.stats.kurtosis(agg_performance.returns)])

        if any([field is not None for field in (
            agg_performance.abs_exposures,
            agg_performance.net_exposures,
            agg_performance.total_holdings,
            agg_performance.turnover
            )]):
            stats.append(["", " Positions and Exposure"])

        if agg_performance.abs_exposures is not None:
            avg_abs_exposures = agg_performance.abs_exposures.mean()
            stats.append([
                "Absolute Exposure (percentage of capital)",
                "{0}%".format(round(avg_abs_exposures * 100, 1))])

        if agg_performance.net_exposures is not None:
            avg_net_exposures = agg_performance.net_exposures.mean()
            stats.append([
                "Net Exposure (percentage of capital)",
                "{0}%".format(round(avg_net_exposures * 100, 1))])

        if agg_performance.total_holdings is not None:
            avg_daily_holdings = agg_performance.total_holdings.mean()
            stats.append([
                "Average Daily Holdings",
                round(avg_daily_holdings)])

        if agg_performance.turnover is not None:
            avg_daily_turnover = agg_performance.turnover.mean()
            stats.append([
                "Average Daily Turnover (percentage of capital)",
                "{0}%".format(round(avg_daily_turnover * 100, 1))])

        if agg_performance.abs_exposures is not None:
            norm_cagr = agg_performance.cagr / avg_abs_exposures
            stats.append([
                "Normalized CAGR (CAGR/Absolute Exposure)",
                "{0}%".format(round(norm_cagr * 100, 1))])

        with sns.axes_style("white"):

            fig = plt.figure("Performance Summary", figsize=(6,6))

            axis = fig.add_subplot(111)
            axis.axis("off")

            headings, values = zip(*stats)

            table = axis.table(
                cellText=[[v] for v in values],
                rowLabels=headings,
                colLabels=["Performance Summary"],
                loc="center"
            )
            for (row, col), cell in table.get_celld().items():
                txt = cell.get_text().get_text()
                if row == 0 or txt.startswith(" "):
                    cell.set_text_props(fontproperties=FontProperties(weight='bold'))

            table.scale(1, 2)
            table.set_fontsize("large")

    def _create_constituents_tearsheet(self, performance):
        """
        Create a tear sheet of the strategies or symbols in the data.
        """
        with sns.axes_style("white"):

            fig = plt.figure("Strategies or Securities", figsize=(6,6))

            axis = fig.add_subplot(111)
            axis.axis("off")

            cols = list(performance.returns.columns)
            if len(cols) > 58:
                hidden_cols = len(cols) - 58
                cols = cols[0:58]
                cols.append("and {0} more".format(hidden_cols))

            cells_per_row = 4
            cells = ["Included columns:"] + cols
            num_cells = len(cells)
            if num_cells > cells_per_row and num_cells % cells_per_row != 0:
                # Cells must be divisible by cells_per_row for matplotlib table
                extra_cells_required = cells_per_row - num_cells % cells_per_row
                for _ in range(extra_cells_required):
                    cells.append("")

            table = axis.table(
                cellText=[cells[i:i + cells_per_row] for i in range(0, len(cells), cells_per_row)],
                loc="top"
            )
            for (row, col), cell in table.get_celld().items():
                if (row == 0) and (col == 0):
                    cell.set_text_props(fontproperties=FontProperties(weight='bold'))

            table.scale(2, 2)
            table.set_fontsize("large")

    def create_returns_tearsheet(self, performance, agg_performance=None):
        """
        Create a tear sheet of returns-related plots.

        The included plots depend on what is present in the performance data.
        Always plots cumulative returns, drawdowns, and rolling Sharpe. Plots
        cumulative returns vs benchmark if benchmark is present. Plots
        cumulative PNL if PNL is present. For multi-column performance
        data (multi-strategy or detailed single-strategy), plots bar
        charts of Sharpe, CAGR, and PNL if present.

        Parameters
        ----------
        performance : DailyPerformance, required
            a DailyPerformance instance

        agg_performance : AggregateDailyPerformance, optional
            an AggregateDailyPerformance instance. Constructed from performance
            if not provided.

        Returns
        -------
        None

        Examples
        --------
        >>> from moonchart import DailyPerformance, Tearsheet
        >>> perf = DailyPerformance.from_moonshot_csv("backtest_results.csv")
        >>> t = Tearsheet()
        >>> t.create_returns_tearsheet(perf)
        """
        if agg_performance is None:
            agg_performance = AggregateDailyPerformance(performance)

        num_cols = len(performance.returns.columns)
        show_details = num_cols > 1 and num_cols <= self.max_cols_for_details

        width, height = self.figsize
        # cut height in half if not showing details
        if not show_details:
            height /= 2

        self._create_returns_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "",
            figsize=(width, height))

        if show_details:
            self._create_returns_plots(performance, subplot=212, extra_label="(Details)")
            self._create_detailed_returns_bar_charts(performance)

    def _create_detailed_returns_bar_charts(self, performance):

        fig = plt.figure("Returns (Details)", figsize=self.figsize)

        color_palette = sns.color_palette()
        num_series = len(performance.cum_returns.columns)
        if num_series > 6:
            color_palette = sns.color_palette("hls", num_series)

        with sns.color_palette(color_palette):

            axis = fig.add_subplot(2,2,1)
            axis.set_ylabel("CAGR")
            self._y_format_as_percentage(axis)
            cagr = performance.cagr.copy()
            cagr.index = cagr.index.astype(str).str.wrap(10)
            cagr.plot(ax=axis, kind="bar", title="CAGR (Details)")

            axis = fig.add_subplot(2,2,2)
            self._y_format_at_least_two_decimal_places(axis)
            axis.set_ylabel("Sharpe ratio")
            sharpe = performance.sharpe.copy()
            sharpe.index = sharpe.index.astype(str).str.wrap(10)
            sharpe.plot(ax=axis, kind="bar", title="Sharpe (Details)")

            axis = fig.add_subplot(2,2,3)
            axis.set_ylabel("Drawdown")
            self._y_format_as_percentage(axis)
            max_drawdowns = performance.max_drawdown.copy()
            max_drawdowns.index = max_drawdowns.index.astype(str).str.wrap(10)
            max_drawdowns.plot(ax=axis, kind="bar", title="Max drawdown (Details)")

        fig.tight_layout()

        if performance.pnl is not None:
            fig = plt.figure("PNL (Details)", figsize=self.figsize)
            axis = fig.add_subplot(111)
            axis.set_ylabel("PNL")
            pnl = performance.pnl.sum()
            if performance.commission_amounts is not None:
                pnl.name = "pnl"
                commission_amounts = performance.commission_amounts.sum()
                commission_amounts.name = "commissions"
                gross_pnl = pnl + commission_amounts
                gross_pnl.name = "gross pnl"
                try:
                    pnl = pd.concat((pnl, gross_pnl, commission_amounts), axis=1, sort=True)
                except TypeError:
                    # sort was introduced in pandas 0.23
                    pnl = pd.concat((pnl, gross_pnl, commission_amounts), axis=1)
            pnl.plot(
                ax=axis, kind="bar", title="PNL (Details)")

    def create_positions_tearsheet(self, performance, agg_performance=None):
        """
        Create a tear sheet of position-related plots.

        Includes plots of net and absolute daily exposure, number of daily
        holdings, and daily turnover.

        Parameters
        ----------
        performance : DailyPerformance, required
            a DailyPerformance instance

        agg_performance : AggregateDailyPerformance, optional
            an AggregateDailyPerformance instance. Constructed from performance
            if not provided.

        Returns
        -------
        None

        Examples
        --------
        >>> from moonchart import DailyPerformance, Tearsheet
        >>> perf = DailyPerformance.from_moonshot_csv("backtest_results.csv")
        >>> t = Tearsheet()
        >>> t.create_positions_tearsheet(perf)
        """
        if agg_performance is None:
            agg_performance = AggregateDailyPerformance(performance)

        num_cols = len(performance.returns.columns)
        show_details = num_cols > 1 and num_cols <= self.max_cols_for_details

        width, height = self.figsize
        # cut height in half if not showing details
        if not show_details:
            height /= 2

        self._create_positions_plots(
            agg_performance,
            subplot=211 if show_details else 111,
            extra_label="(Aggregate)" if show_details else "",
            figsize=(width, height))

        if show_details:
            self._create_positions_plots(performance, subplot=212, extra_label="(Details)")
            self._create_detailed_positions_bar_charts(performance)

    def _create_positions_plots(self, performance, subplot, extra_label, figsize=None):

        figsize = figsize or self.figsize

        color_palette = sns.color_palette()

        if isinstance(performance.returns, pd.DataFrame):
            num_series = len(performance.returns.columns)
            if num_series > 6:
                color_palette = sns.color_palette("hls", num_series)

        with sns.color_palette(color_palette):

            if performance.abs_exposures is not None:
                fig = plt.figure("Absolute Exposure", figsize=figsize)
                axis = fig.add_subplot(subplot)
                self._y_format_as_percentage(axis)
                plot = performance.abs_exposures.round(2).plot(ax=axis, title="Absolute Exposure {0}".format(extra_label))
                axis.set_ylabel("Percentage of capital")
                axis.set_xlabel("")
                if isinstance(performance.abs_exposures, pd.DataFrame):
                    self._clear_legend(plot)

            if performance.net_exposures is not None:
                fig = plt.figure("Net Exposure", figsize=figsize)
                axis = fig.add_subplot(subplot)
                self._y_format_as_percentage(axis)
                plot = performance.net_exposures.round(2).plot(ax=axis, title="Net Exposure {0}".format(extra_label))
                axis.set_ylabel("Percentage of capital")
                axis.set_xlabel("")
                if isinstance(performance.net_exposures, pd.DataFrame):
                    self._clear_legend(plot)

            if performance.total_holdings is not None:
                fig = plt.figure("Daily Holdings", figsize=figsize)
                axis = fig.add_subplot(subplot)
                plot = performance.total_holdings.plot(ax=axis, title="Daily Holdings {0}".format(extra_label))
                axis.set_ylabel("Number of holdings")
                axis.set_xlabel("")
                if isinstance(performance.total_holdings, pd.DataFrame):
                    self._clear_legend(plot)

            if performance.turnover is not None:
                fig = plt.figure("Daily Turnover", figsize=figsize)
                axis = fig.add_subplot(subplot)
                self._y_format_as_percentage(axis)
                turnover = performance.turnover
                plot = turnover.plot(ax=axis, title="Daily Turnover {0}".format(extra_label))
                axis.set_ylabel("Percentage of capital")
                axis.set_xlabel("")
                if isinstance(turnover, pd.DataFrame):
                    self._clear_legend(plot)

    def _create_detailed_positions_bar_charts(self, performance):

        # extend figsize due to 3 rows
        width, height = self.figsize
        figsize = width, height*1.5

        fig = plt.figure("Positions (Details)", figsize=figsize)

        color_palette = sns.color_palette()
        num_series = len(performance.cum_returns.columns)
        if num_series > 6:
            color_palette = sns.color_palette("hls", num_series)

        total_plots = sum([1 for field in (
            performance.abs_exposures,
            performance.net_exposures,
            performance.total_holdings,
            performance.turnover,
            performance.abs_exposures) if field is not None])

        rows = math.ceil(total_plots/2)

        with sns.color_palette(color_palette):

            next_pos = 1

            if performance.abs_exposures is not None:
                avg_abs_exposures = performance.abs_exposures.mean()
                axis = fig.add_subplot(rows,2,next_pos)
                next_pos += 1
                self._y_format_as_percentage(axis)
                avg_abs_exposures.plot(ax=axis, kind="bar", title="Avg Absolute Exposure (Details)")
                axis.set_ylabel("Percentage of capital")

            if performance.net_exposures is not None:
                avg_net_exposures = performance.net_exposures.mean()
                axis = fig.add_subplot(rows,2,next_pos)
                next_pos += 1
                self._y_format_as_percentage(axis)
                avg_net_exposures.plot(ax=axis, kind="bar", title="Avg Net Exposure (Details)")
                axis.set_ylabel("Percentage of capital")

            if performance.total_holdings is not None:
                avg_total_holdings = performance.total_holdings.mean()
                axis = fig.add_subplot(rows,2,next_pos)
                next_pos += 1
                avg_total_holdings.plot(ax=axis, kind="bar", title="Avg Daily Holdings (Details)")
                axis.set_ylabel("Number of holdings")

            if performance.turnover is not None:
                avg_daily_turnover = performance.turnover.mean()
                axis = fig.add_subplot(rows,2,next_pos)
                next_pos += 1
                self._y_format_as_percentage(axis)
                avg_daily_turnover.plot(ax=axis, kind="bar", title="Avg Daily Turnover (Details)")
                axis.set_ylabel("Percentage of capital")

            if performance.abs_exposures is not None:
                norm_cagrs = performance.cagr / avg_abs_exposures
                axis = fig.add_subplot(rows,2,next_pos)
                next_pos += 1
                self._y_format_as_percentage(axis)
                norm_cagrs.plot(ax=axis, kind="bar", title="Normalized CAGR (CAGR/Absolute Exposure) (Details)")
                axis.set_ylabel("Normalized CAGR")

        fig.tight_layout()

    def create_returns_by_year_tearsheet(self, performance, agg_performance=None):
        """
        Plots bar charts showing CAGR and Sharpe by year.

        Parameters
        ----------
        performance : DailyPerformance, required
            a DailyPerformance instance

        agg_performance : AggregateDailyPerformance, optional
            an AggregateDailyPerformance instance. Constructed from performance
            if not provided.

        Returns
        -------
        None

        Examples
        --------
        >>> from moonchart import DailyPerformance, Tearsheet
        >>> perf = DailyPerformance.from_moonshot_csv("backtest_results.csv")
        >>> t = Tearsheet()
        >>> t.create_returns_by_year_tearsheet(perf)
        """
        if agg_performance is None:
            agg_performance = AggregateDailyPerformance(performance)

        num_cols = len(performance.returns.columns)
        show_details = num_cols > 1 and num_cols <= self.max_cols_for_details

        width, height = self.figsize
        # cut height in half if not showing details
        if not show_details:
            height /= 2

        fig = plt.figure("Returns by Year", figsize=(width, height))

        self._create_returns_by_year_plots(
            agg_performance,
            rows=2 if show_details else 1,
            row=1,
            fig=fig,
            extra_label="(Aggregate)" if show_details else "")

        if show_details:
            self._create_returns_by_year_plots(performance, rows=2, row=2, fig=fig, extra_label="(Details)")

    def _create_returns_by_year_plots(self, performance, rows, row, fig, extra_label):

        color_palette = sns.color_palette()

        if isinstance(performance.returns, pd.DataFrame):
            num_series = len(performance.cum_returns.columns)
            if num_series > 6:
                color_palette = sns.color_palette("hls", num_series)
        else:
            color_palette = sns.color_palette()[0:1]

        grouped_returns = performance.returns.groupby(performance.returns.index.year)
        cagrs_by_year = grouped_returns.apply(lambda x: get_cagr(
            get_cum_returns(x, compound=performance.compound),
            compound=performance.compound))
        sharpes_by_year = grouped_returns.apply(get_sharpe, riskfree=performance.riskfree)

        cols = 2
        # 2 cols per row, minus 1, gives the start position
        start_at = 2 * row - 1
        with sns.color_palette(color_palette):
            axis = fig.add_subplot(rows, 2, start_at)
            axis.set_ylabel("CAGR")
            self._y_format_as_percentage(axis)
            plot = cagrs_by_year.plot(ax=axis, kind="bar", title="CAGR by Year {0}".format(extra_label))
            axis.set_xlabel("")
            if isinstance(cagrs_by_year, pd.DataFrame):
                # Remove legend, rely on legend from Sharpe plot
                plot.legend_.remove()

            axis = fig.add_subplot(rows, 2, start_at+1)
            axis.set_ylabel("Sharpe ratio")
            self._y_format_at_least_two_decimal_places(axis)
            plot = sharpes_by_year.plot(ax=axis, kind="bar", title="Sharpe by Year {0}".format(extra_label))
            axis.set_xlabel("")
            if isinstance(sharpes_by_year, pd.DataFrame):
                self._clear_legend(plot)

        fig.tight_layout()

    def create_montecarlo_tearsheet(self, performance, cycles=5, aggregate_before_shuffle=True):
        """
        Run a Montecarlo simulation by shuffling the DataFrame of returns a specified
        number of times and plotting the shuffled returns against the original returns.

        Parameters
        ----------
        performance : DailyPerformance, required
            a DailyPerformance instance

        cycles : int, optional
            the number of Montecarlo simulations (default 5)

        aggregate_before_shuffle : bool
            whether to aggregate daily returns before or after shuffling. Only relevant to
            multi-column (that is, multi-strategy or detailed single-strategy) DataFrames.
            If True, aggregated daily returns are preserved and only the order of days is
            randomized. If False, each column's returns are shuffled separately, without
            preservation of daily aggregations. False is more random. True may be preferable
            if daily returns across columns are expected to be correlated. Default True.

        Returns
        -------
        None

        Examples
        --------
        >>> from moonchart import DailyPerformance, Tearsheet
        >>> perf = DailyPerformance.from_moonshot_csv("backtest_results.csv")
        >>> t = Tearsheet()
        >>> t.create_montecarlo_tearsheet(perf, cycles=10)
        """
        all_simulations = []

        returns = performance.returns

        if aggregate_before_shuffle:
            returns = returns.sum(axis=1)

        for i in range(cycles):
            if aggregate_before_shuffle:
                sim_returns = pd.Series(np.random.permutation(returns), index=returns.index)
            else:
                sim_returns = returns.apply(np.random.permutation).sum(axis=1)
            all_simulations.append(sim_returns)

        try:
            sim_returns = pd.concat(all_simulations, axis=1, sort=False)
        except TypeError:
            # sort was introduced in pandas 0.23
            sim_returns = pd.concat(all_simulations, axis=1)

        if not aggregate_before_shuffle:
            returns = returns.sum(axis=1)

        cum_sim_returns = get_cum_returns(sim_returns, compound=performance.compound)
        cum_returns = get_cum_returns(returns, compound=performance.compound)
        fig = plt.figure("Montecarlo Simulation", figsize=self.figsize)
        fig.suptitle(self._suptitle, **self._suptitle_kwargs)
        axis = fig.add_subplot(211)
        self._y_format_at_least_two_decimal_places(axis)
        axis.set_ylabel("Cumulative return")
        with_baseline(cum_sim_returns).plot(ax=axis, title="Montecarlo Cumulative Returns ({0} cycles)".format(cycles), legend=False)
        with_baseline(cum_returns).plot(ax=axis, linewidth=4, color="black")
        axis = fig.add_subplot(212)
        self._y_format_as_percentage(axis)
        axis.set_ylabel("Drawdown")
        get_drawdowns(cum_sim_returns).plot(ax=axis, title="Montecarlo Drawdowns ({0} cycles)".format(cycles), legend=False)
        get_drawdowns(cum_returns).plot(ax=axis, linewidth=4, color="black")
