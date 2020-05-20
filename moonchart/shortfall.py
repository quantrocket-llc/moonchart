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
from matplotlib.font_manager import FontProperties
import seaborn as sns
import empyrical as ep
import scipy.stats
import warnings
from .base import BaseTearsheet
from .perf import DailyPerformance, AggregateDailyPerformance
from .exceptions import MoonchartError
from .utils import with_baseline, get_cum_returns, get_drawdowns

class ShortfallTearsheet(BaseTearsheet):
    """
    Generate a tear sheet of performance stats and plots highlighting the
    shortfall between simulated or benchmark results and actual results.

    See Also
    --------
    ShortfallTearsheet.from_csvs : Create a shortfall tear sheet from CSVs.
    """

    def __init__(self,
                 *args,
                 labels=("simulated", "actual"),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.x_label, self.y_label = labels

    @classmethod
    def from_csvs(cls, x_filepath_or_buffer, y_filepath_or_buffer,
                  labels=("simulated", "actual"),
                  start_date=None,
                  end_date=None,
                  largest_n=None,
                  shift_x_positions=0,
                  figsize=None, trim_outliers=None,
                  how_to_aggregate=None,
                  pdf_filename=None, riskfree=0,
                  compound=True, rolling_sharpe_window=200):
        """
        Create a shortfall tear sheet comparing simulated results (or other benchmark results)
        with actual results.

        Parameters
        ----------
        x_filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV containing simulated results or other
            benchmark results, usually a Moonshot backtest CSV

        y_filepath_or_buffer : str or file-like object
            filepath or file-like object of the CSV containing actual results, usually
            a PNL CSV from the blotter

        labels : tuple, optional
            labels for the x and y results, default ('simulated', 'actual')

        start_date : str (YYYY-MM-DD), optional
            truncate at this start date (otherwise include entire date range)

        end_date : str (YYYY-MM-DD), optional
            truncate at this end date (otherwise include entire date range)

        largest_n : int, optional
            include a "largest shortfalls" table with this many records highligting
            the dates and specific instruments or strategies (depending on whether
            the columns of the input CSVs are instruments or strategies) with the
            largest magnitude shortfall (positive or negative). Also results in
            additional plots depicting performance exluding the largest shortfalls.
            Default is None, meaning don't include a largest shortfalls table or
            additional plots.

        shift_x_positions : int, optional
            shift positions forward (positive integer) or backward (negative integer)
            in simulated/benchmark results. Can be used to align simulated positions
            with actual positions.

        figsize : tuple (width, height), optional
            (width, height) of matplotlib figure. Default is (16, 12)

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
        DataFrame or None
            DataFrame of largest shortfalls if largest_n is not None, otherwise None

        Examples
        --------
        >>> from moonshot import ShortfallTearsheet
        >>> ShortfallTearsheet.from_csvs("backtest.csv", "pnl.csv")
        """

        x_perf = DailyPerformance.from_moonshot_csv(
            x_filepath_or_buffer,
            start_date=start_date,
            end_date=end_date,
            trim_outliers=trim_outliers,
            how_to_aggregate=how_to_aggregate,
            riskfree=riskfree,
            compound=compound,
            rolling_sharpe_window=rolling_sharpe_window)

        y_perf = DailyPerformance.from_pnl_csv(
            y_filepath_or_buffer,
            start_date=start_date,
            end_date=end_date,
            trim_outliers=trim_outliers,
            how_to_aggregate=how_to_aggregate,
            riskfree=riskfree,
            compound=compound,
            rolling_sharpe_window=rolling_sharpe_window)

        t = cls(labels=labels, figsize=figsize, pdf_filename=pdf_filename)
        return t.create_full_tearsheet(x_perf, y_perf, largest_n=largest_n,
                                       shift_x_positions=shift_x_positions)

    def create_full_tearsheet(self, x_performance, y_performance, largest_n=None,
                              shift_x_positions=0):
        """
        Create a shortfall tear sheet comparing simulated results (or other
        benchmark results) with actual results.

        Parameters
        ----------
        x_performance : DailyPerformance, required
            simulated or benchmark performance results

        y_performance : DailyPerformance, required
            actual performance results

        largest_n : int, optional
            include a "largest shortfalls" table with this many records highligting
            the dates and specific instruments or strategies (depending on whether
            the columns of the input CSVs are instruments or strategies) with the
            largest magnitude shortfall (positive or negative). Also results in
            additional plots depicting performance exluding the largest shortfalls.
            Default is None, meaning don't include a largest shortfalls table or
            additional plots.

        shift_x_positions : int, optional
            shift positions forward (positive integer) or backward (negative integer)
            in simulated/benchmark results. Can be used to align simulated positions
            with actual positions.

        Returns
        -------
        DataFrame or None
            DataFrame of largest shortfalls if largest_n is not None, otherwise None

        See Also
        --------
        ShortfallTearsheet.from_csvs : Create a shortfall tear sheet from CSVs.
        """

        y_agg_performance = AggregateDailyPerformance(y_performance)
        x_agg_performance = AggregateDailyPerformance(x_performance)

        self.create_summary_tearsheet(x_agg_performance, y_agg_performance)

        largest_shortfalls = None
        if largest_n:
            largest_shortfalls = self.create_largest_shortfalls_tearsheet(x_performance, y_performance, largest_n=largest_n)

        # Returns
        self._plot_field(
            x_agg_performance.returns,
            y_agg_performance.returns,
            fig_title="Returns",
            cumplots=True
        )

        if largest_shortfalls is not None:
            largest_shortfalls_x_mask = largest_shortfalls.shortfall.unstack().reindex(
                index=x_performance.returns.index,
                columns=x_performance.returns.columns)
            largest_shortfalls_y_mask = largest_shortfalls.shortfall.unstack().reindex(
                index=y_performance.returns.index,
                columns=y_performance.returns.columns)

            # Returns excluding largest shortfalls
            self._plot_field(
                x_performance.returns.where(largest_shortfalls_x_mask.isnull(), 0).sum(axis=1),
                y_performance.returns.where(largest_shortfalls_y_mask.isnull(), 0).sum(axis=1),
                fig_title="Returns",
                fig_title_suffix=" excluding Top {} Largest Shortfalls".format(largest_n),
                cumplots=True
            )

        # Commissions
        if x_agg_performance.commissions is not None and y_agg_performance.commissions is not None:
            self._plot_field(
                x_agg_performance.commissions,
                y_agg_performance.commissions,
                fig_title="Commissions",
                format_as_pct_decimal_places=2)

        # Total holdings
        if x_agg_performance.total_holdings is not None and y_agg_performance.total_holdings is not None:
            self._plot_field(
                x_agg_performance.total_holdings.shift(shift_x_positions),
                y_agg_performance.total_holdings,
                fig_title="Total Holdings",
                format_as_pct_decimal_places=None)

        # Net exposures
        if x_agg_performance.net_exposures is not None and y_agg_performance.net_exposures is not None:
            self._plot_field(
                x_agg_performance.net_exposures.shift(shift_x_positions),
                y_agg_performance.net_exposures,
                fig_title="Net Exposure")

        # Absolute exposures
        if x_agg_performance.abs_exposures is not None and y_agg_performance.abs_exposures is not None:
            self._plot_field(
                x_agg_performance.abs_exposures.shift(shift_x_positions),
                y_agg_performance.abs_exposures,
                fig_title="Absolute Exposure")

        # Turnover
        if x_agg_performance.turnover is not None and y_agg_performance.turnover is not None:
            self._plot_field(
                x_agg_performance.turnover.shift(shift_x_positions),
                y_agg_performance.turnover,
                fig_title="Turnover")

        self._save_or_show()

        return largest_shortfalls

    def create_largest_shortfalls_tearsheet(self, x_performance, y_performance, largest_n=5):
        """
        Create a "largest shortfalls" table highligting the dates and
        specific instruments or strategies (depending on whether the columns
        of the input returns are instruments or strategies) with the largest
        magnitude shortfall (positive or negative).

        Parameters
        ----------
        x_performance : DailyPerformance, required
            simulated or benchmark performance results

        y_performance : DailyPerformance, required
            actual performance results

        largest_n : int, optional
            the number of comparisons to include in the table.

        Returns
        -------
        DataFrame
            DataFrame of largest shortfalls
        """

        shared_columns = x_performance.returns.columns.intersection(y_performance.returns.columns)
        if shared_columns.empty:
            warnings.warn("skipping largest shortfalls table because {x} and {y} have no shared columns".format(
                x=self.x_label, y=self.y_label))
            return

        x_returns = x_performance.returns.stack()
        y_returns = y_performance.returns.stack()

        x_returns.name = self.x_label
        y_returns.name = self.y_label

        returns = pd.concat([x_returns, y_returns], axis=1).fillna(0)

        returns["shortfall"] = returns[self.y_label] - returns[self.x_label]
        returns["abs_shortfall"] = returns.shortfall.abs()

        returns = returns[returns.abs_shortfall > 0]
        if returns.empty:
            warnings.warn("skipping largest shortfalls table because no shortfall found")
            return None

        largest_shortfalls = returns.sort_values(
            "abs_shortfall", ascending=False).drop(
                "abs_shortfall", axis=1).head(largest_n)

        largest_shortfalls_raw = largest_shortfalls.copy()

        largest_shortfalls.index = largest_shortfalls.index.set_names(["Date", "Sid"])
        # format as pct
        largest_shortfalls = (largest_shortfalls * 100).round(2).astype(str) + "%"

        largest_shortfalls = largest_shortfalls.reset_index()
        largest_shortfalls.insert(
            0,
            "Top {} Largest Shortfalls".format(largest_n),
            largest_shortfalls.Date.dt.strftime("%Y-%m-%d").str.cat(largest_shortfalls.Sid, sep=" "))

        largest_shortfalls = largest_shortfalls.drop(["Date","Sid"], axis=1)

        with sns.axes_style("white"):

            fig = plt.figure("Top Shortfalls", figsize=(6,6))

            axis = fig.add_subplot(111)
            axis.axis("off")

            largest_shortfalls_dict = largest_shortfalls.to_dict("split")

            table = axis.table(
                colWidths=[0.4, 0.2, 0.2, 0.2],
                cellText=[largest_shortfalls_dict["columns"]] + largest_shortfalls_dict["data"],
                loc="center"
            )

            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontproperties=FontProperties(weight='bold'))

            table.scale(2, 2)
            table.set_fontsize("large")

        return largest_shortfalls_raw

    def _plot_field(self, x, y, fig_title,
                    fig_title_suffix="",
                    format_as_pct_decimal_places=1,
                    cumplots=False):

        y.name = self.y_label
        x.name = self.x_label

        try:
            results = pd.concat((y, x), axis=1, sort=True).fillna(0)
        except TypeError:
            # sort was introduced in pandas 0.23
            results = pd.concat((y, x), axis=1).fillna(0)

        fig = None

        if cumplots:

            cum_results = get_cum_returns(results)
            cum_shortfall = cum_results[self.y_label] - cum_results[self.x_label]

            cum_shortfall = pd.concat([
                cum_shortfall.where(cum_shortfall>0),
                cum_shortfall.where(cum_shortfall<0)
                ], axis=1, keys=["positive", "negative"])

            fig, (ax1, ax2, axis) = plt.subplots(
                3, 1, sharex=True, gridspec_kw={
                    "height_ratios":[0.5,0.25,0.25],
                    'hspace': 0.1},
                num="Cumulative " + fig_title + fig_title_suffix, figsize=self.figsize)
            with_baseline(cum_results).plot(ax=ax1, title="Cumulative " + fig_title + fig_title_suffix)
            self._y_format_at_least_two_decimal_places(ax1)
            ax1.set_ylabel("Cumulative " + fig_title)
            ax1.set_xlabel("")

            ax2.set_ylabel("Shortfall")
            self._y_format_as_percentage(ax2, max_decimal_places=2)
            ax2.set_ylabel("Percentage of capital")
            with_baseline(cum_shortfall, value=0).plot(
                ax=ax2, title="Cumulative Shortfall", kind="area",
                stacked=False,
                legend=False,
                color=["C1","C2"] # 2nd and 3rd color in palette are green and red
            )
            ax2.set_xlabel("")

        if fig is None:
            width, height = self.figsize
            fig = plt.figure(fig_title + fig_title_suffix, figsize=(width, height/4))
            axis = fig.add_subplot(111)

        if format_as_pct_decimal_places is not None:
            self._y_format_as_percentage(axis, max_decimal_places=format_as_pct_decimal_places)
            axis.set_ylabel("Percentage of capital")

        shortfall = results[self.y_label] - results[self.x_label]
        results["shortfall"] = shortfall
        results = with_baseline(results, value=0)
        results.plot(ax=axis, title="Daily " + fig_title)
        axis.set_xlabel("")

    def create_summary_tearsheet(self, x_performance, y_performance):
        """
        Create a tear sheet of summary shortfall stats in a table.

        Parameters
        ----------
        x_performance : AggregateDailyPerformance, required
            simulated or benchmark aggregate performance results

        y_performance : AggregateDailyPerformance, required
            actual aggregate performance results

        Returns
        -------
        None

        """
        stats = []

        stats.append(
            ["Start Date",
             [x_performance.returns.index.min().date().isoformat(),
              y_performance.returns.index.min().date().isoformat()]])
        stats.append(
            ["End Date",
             [x_performance.returns.index.max().date().isoformat(),
              y_performance.returns.index.max().date().isoformat()]])

        stats.append(
            ['Total Months',
             [round((x_performance.returns.index.max() - x_performance.returns.index.min()) / pd.Timedelta(365.25/12, 'D')),
              round((y_performance.returns.index.max() - y_performance.returns.index.min()) / pd.Timedelta(365.25/12, 'D'))]])

        stats.append([" Risk and Returns", ["", ""]])
        stats.append([
            "CAGR",
            ["{0}%".format(round(x_performance.cagr * 100, 1)),
             "{0}%".format(round(y_performance.cagr * 100, 1))]])
        stats.append([
            "Sharpe Ratio",
            ['%.2f' % x_performance.sharpe,
             '%.2f' % y_performance.sharpe]])
        stats.append([
            "Max Drawdown",
            ["{0}%".format(round(x_performance.max_drawdown * 100, 1)),
             "{0}%".format(round(y_performance.max_drawdown * 100, 1))]])
        stats.append([
            "Cumulative Return",
            ["{0}%".format(round(ep.cum_returns_final(x_performance.returns) * 100, 1)),
             "{0}%".format(round(ep.cum_returns_final(y_performance.returns) * 100, 1))]])
        if x_performance.commissions is not None and y_performance.commissions is not None:
            stats.append([
                "Cumulative Commissions",
                ["{0}%".format(round(ep.cum_returns_final(x_performance.commissions) * 100, 1)),
                 "{0}%".format(round(ep.cum_returns_final(y_performance.commissions) * 100, 1))]])
        stats.append([
            "Annual Volatility",
            ["{0}%".format(round(ep.annual_volatility(x_performance.returns) * 100, 1)),
             "{0}%".format(round(ep.annual_volatility(y_performance.returns) * 100, 1))]])
        stats.append([
            "Sortino Ratio",
            ['%.2f' % ep.sortino_ratio(x_performance.returns),
             '%.2f' % ep.sortino_ratio(y_performance.returns)]])
        stats.append([
            "Calmar Ratio",
            ['%.2f' % ep.calmar_ratio(x_performance.returns),
             '%.2f' % ep.calmar_ratio(y_performance.returns)]])
        stats.append([
            "Skew",
            ['%.2f' % scipy.stats.skew(x_performance.returns),
             '%.2f' % scipy.stats.skew(y_performance.returns)]])
        stats.append([
            "Kurtosis",
            ['%.2f' % scipy.stats.kurtosis(x_performance.returns),
             '%.2f' % scipy.stats.kurtosis(y_performance.returns)]])

        stats.append([" Positions and Exposure", ["", ""]])

        if x_performance.abs_exposures is not None and y_performance.abs_exposures is not None:
            stats.append([
                "Absolute Exposure (percentage of capital)",
                ["{0}%".format(round(x_performance.abs_exposures.mean() * 100, 1)),
                 "{0}%".format(round(y_performance.abs_exposures.mean() * 100, 1))]])

        if x_performance.net_exposures is not None and y_performance.net_exposures is not None:
            stats.append([
                "Net Exposure (percentage of capital)",
                ["{0}%".format(round(x_performance.net_exposures.mean() * 100, 1)),
                 "{0}%".format(round(y_performance.net_exposures.mean() * 100, 1))]])

        if x_performance.total_holdings is not None and y_performance.total_holdings is not None:
            stats.append([
                "Average Daily Holdings",
                [round(x_performance.total_holdings.mean()),
                 round(y_performance.total_holdings.mean())]])

        if x_performance.turnover is not None and y_performance.turnover is not None:
            stats.append([
                "Average Daily Turnover (percentage of capital)",
                ["{0}%".format(round(x_performance.turnover.mean() * 100, 1)),
                 "{0}%".format(round(y_performance.turnover.mean() * 100, 1))]])

        if x_performance.abs_exposures is not None and y_performance.abs_exposures is not None:
            stats.append([
                "Normalized CAGR (CAGR/Absolute Exposure)",
                ["{0}%".format(round((x_performance.cagr / x_performance.abs_exposures.mean()) * 100, 1)),
                 "{0}%".format(round((y_performance.cagr / y_performance.abs_exposures.mean()) * 100, 1))]])

        with sns.axes_style("white"):

            fig = plt.figure("Performance Summary", figsize=(6,6))

            axis = fig.add_subplot(111)
            axis.axis("off")

            headings, values = zip(*stats)

            table = axis.table(
                cellText=values,
                rowLabels=headings,
                colLabels=[self.x_label, self.y_label],
                loc="center"
            )
            for (row, col), cell in table.get_celld().items():
                txt = cell.get_text().get_text()
                if row == 0 or txt.startswith(" "):
                    cell.set_text_props(fontproperties=FontProperties(weight='bold'))

            table.scale(1, 2)
            table.set_fontsize("large")
