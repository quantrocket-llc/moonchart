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

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cycler
from matplotlib.backends.backend_pdf import PdfPages
from .utils import (
    with_baseline,
    get_cum_returns)
from matplotlib.ticker import FuncFormatter

class BaseTearsheet(object):
    """
    Base class for tear sheets.

    Parameters
    ----------
    figsize : tuple (width, height), optional
        (width, height) of matplotlib figure. Default is (16, 12)

    pdf_filename : string, optional
        save tear sheet to this filepath as a PDF instead of displaying

    max_cols_for_details : int, optional
        suppress detailed plots if there are more than this many columns
        (i.e. strategies or securities). Too many plots may cause slow
        rendering. Default 25.
    """
    def __init__(self, pdf_filename=None, figsize=None, max_cols_for_details=25):
        self.figsize = figsize or (16.0, 12.0) # width, height in inches
        plt.rc("axes", axisbelow=True)
        if pdf_filename:
            self._pdf = PdfPages(pdf_filename, keep_empty=True)
        else:
            self._pdf = None

        self._suptitle = None
        self._suptitle_kwargs = {
            "bbox": dict(facecolor="#EAEAF2", edgecolor='white', alpha=0.5)}
        self.max_cols_for_details = max_cols_for_details

    def _save_or_show(self):
        """
        Saves the fig to the multi-page PDF, or shows it.
        """
        if self._pdf:
            for fignum in plt.get_fignums():
                self._pdf.savefig(fignum, bbox_inches='tight')
            plt.close("all")
            self._pdf.close()
        else:
            plt.show()

    def _y_format_as_percentage(self, axis, max_decimal_places=1):
        """
        Sets a Y-axis formatter that converts a decimal to a percentage (e.g.
        0.12 -> 12.0%)
        """
        def format_as_pct(x, pos):
            # Round to max_decimal_places (12.1%) unless it doesn't matter (12%
            # not 12.0%)
            decimal_places = max_decimal_places

            while decimal_places > 0:
                rounded_result = round(x, decimal_places+2)
                more_rounded_result = round(x, decimal_places+1)

                if rounded_result != more_rounded_result:
                    return ('{:.%d%%}' % decimal_places).format(x)

                decimal_places -= 1

            return '{:.0%}'.format(x)

        y_axis_formatter = FuncFormatter(format_as_pct)
        axis.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    def _y_format_at_least_two_decimal_places(self, axis):
        """
        Sets a Y-axis formatter that rounds a decimal to two decimal places.
        """
        def format_at_least_two_decimal_places(x, pos):
            if round(x,2) == round(x,3):
                return '{:.2f}'.format(x)
            else:
                decimal_places = 3
                while True:
                    if math.isclose(round(x, decimal_places), round(x, decimal_places+1)):
                        return round(x, decimal_places)
                    decimal_places += 1

        y_axis_formatter = FuncFormatter(format_at_least_two_decimal_places)
        axis.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    def _get_plot_dimensions(self, plot_count):
        """
        Returns a tuple of rows, cols needed to accomodate the plot_count.
        """
        rows = math.ceil(math.sqrt(plot_count))
        cols = math.ceil(plot_count/rows)
        return rows, cols

    def _clear_legend(self, plot, legend_title=None):
        """
        Anchors the legend to the outside right of the plot area so you can
        see the plot.
        """
        plot.legend(
            loc='center left', bbox_to_anchor=(1, 0.5), title=legend_title)

    def _create_returns_plots(self, performance, subplot, extra_label, figsize=None,
                              legend_title=None):
        """
        Creates agg/details plots for cumulative returns, drawdowns, rolling
        Sharpe, and possibly pnl.
        """
        figsize = figsize or self.figsize

        color_palette = sns.color_palette()

        if isinstance(performance.cum_returns, pd.DataFrame):
            num_series = len(performance.cum_returns.columns)
            if performance.benchmark_returns is not None:
                num_series += 1
            if num_series > 6:
                color_palette = sns.color_palette("hls", num_series)

        with sns.color_palette(color_palette):
            fig = plt.figure("Cumulative Returns", figsize=figsize)
            axis = fig.add_subplot(subplot)
            max_return = performance.cum_returns.max(axis=0)
            if isinstance(max_return, pd.Series):
                max_return = max_return.max(axis=0)
            # If the price more than doubled, use a log scale
            if max_return >= 2:
                axis.set_yscale("log", basey=2)
                axis.set_ylabel("Cumulative return (log scale)")
            else:
                self._y_format_at_least_two_decimal_places(axis)
                axis.set_ylabel("Cumulative return")

            include_commissions = (
                performance.commissions is not None
                # if all commissions are null/0, don't show them
                and (performance.commissions.fillna(0) != 0).any())

            include_slippage = (
                performance.slippages is not None
                # if all slippages are null/0, don't show them
                and (performance.slippages.fillna(0) != 0).any())

            if (
                # a 212 subplot means a detailed plot, which isn't compatible with
                # showing commissions and slippage
                subplot != 212 and (include_commissions or include_slippage)):

                if include_commissions:
                    cum_commissions = performance.cum_commissions
                    cum_commissions.name = "commissions"

                if include_slippage:
                    cum_slippages = performance.cum_slippages
                    cum_slippages.name = "slippage"

                performance.cum_returns.name = "returns"

                cum_gross_returns = performance.cum_returns

                if include_commissions:
                    cum_gross_returns = cum_gross_returns.multiply(cum_commissions)

                if include_slippage:
                    cum_gross_returns = cum_gross_returns.multiply(cum_slippages)

                cum_gross_returns.name = "gross returns"
                breakdown_parts = [performance.cum_returns, cum_gross_returns]

                if include_commissions:
                    breakdown_parts.append(cum_commissions)

                if include_slippage:
                    breakdown_parts.append(cum_slippages)

                try:
                    returns_breakdown = pd.concat(breakdown_parts, axis=1, sort=True)
                except TypeError:
                    # sort was introduced in pandas 0.23
                    returns_breakdown = pd.concat(breakdown_parts, axis=1)
                plot = with_baseline(returns_breakdown).plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
                axis.set_xlabel("")
                if isinstance(returns_breakdown, pd.DataFrame):
                    self._clear_legend(plot, legend_title=legend_title)
            else:
                plot = with_baseline(performance.cum_returns).plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
                axis.set_xlabel("")
                if isinstance(performance.cum_returns, pd.DataFrame):
                    self._clear_legend(plot, legend_title=legend_title)

            fig = plt.figure("Drawdowns", figsize=figsize)
            axis = fig.add_subplot(subplot)
            axis.set_ylabel("Drawdown")
            self._y_format_as_percentage(axis)
            plot = performance.drawdowns.plot(ax=axis, title="Drawdowns {0}".format(extra_label))
            axis.set_xlabel("")
            if isinstance(performance.drawdowns, pd.DataFrame):
                self._clear_legend(plot, legend_title=legend_title)

            if performance.cum_pnl is not None:
                # Cumulative pnl will generally be very similar to cumulative
                # return, but they might be different if cumulative return is
                # calcuated based on an account balance that changes due to
                # causes other than the strategies being charted (e.g. other
                # strategies, contributions/withdrawals, etc.)
                fig = plt.figure("Cumulative PNL", figsize=figsize)
                axis = fig.add_subplot(subplot)
                axis.set_ylabel("PNL")
                if (
                    performance.commission_amounts is not None
                    # a 212 subplot means a detailed plot, which isn't compatible with
                    # showing commissions
                    and subplot != 212
                    # if all commissions are null/0, don't show them
                    and (performance.commission_amounts.fillna(0) != 0).any()):
                    cum_commissions = performance.cum_commission_amounts
                    cum_commissions.name = "commissions"
                    cum_pnl = performance.cum_pnl
                    cum_pnl.name = "pnl"
                    cum_gross_pnl = cum_pnl + cum_commissions.abs()
                    cum_gross_pnl.name = "gross pnl"
                    try:
                        pnl_breakdown = pd.concat((cum_pnl, cum_gross_pnl, cum_commissions), axis=1, sort=True)
                    except TypeError:
                        # sort was introduced in pandas 0.23
                        pnl_breakdown = pd.concat((cum_pnl, cum_gross_pnl, cum_commissions), axis=1)
                    plot = pnl_breakdown.plot(ax=axis, title="Cumulative PNL {0}".format(extra_label))
                    axis.set_xlabel("")
                    if isinstance(returns_breakdown, pd.DataFrame):
                        self._clear_legend(plot, legend_title=legend_title)
                else:
                    plot = performance.cum_pnl.plot(ax=axis, title="Cumulative PNL {0}".format(extra_label))
                    axis.set_xlabel("")
                    if isinstance(performance.cum_pnl, pd.DataFrame):
                        self._clear_legend(plot, legend_title=legend_title)

            if len(performance.rolling_sharpe.index) > performance.rolling_sharpe_window:
                fig = plt.figure("Rolling Sharpe", figsize=figsize)
                axis = fig.add_subplot(subplot)
                axis.set_ylabel("Sharpe ratio")
                self._y_format_at_least_two_decimal_places(axis)
                plot = performance.rolling_sharpe.plot(ax=axis, title="Rolling Sharpe ({0}-day) {1}".format(
                    performance.rolling_sharpe_window, extra_label))
                axis.set_xlabel("")
                if isinstance(performance.rolling_sharpe, pd.DataFrame):
                    self._clear_legend(plot, legend_title=legend_title)

            if performance.benchmark_returns is not None:
                fig = plt.figure("Cumulative Returns vs Benchmark", figsize=figsize)
                axis = fig.add_subplot(subplot)
                max_return = performance.cum_returns.max(axis=0)
                if isinstance(max_return, pd.Series):
                    max_return = max_return.max(axis=0)

                # If the price more than doubled, use a log scale
                if max_return >= 2:
                    axis.set_yscale("log", basey=2)
                    axis.set_ylabel("Cumulative return (log scale)")
                else:
                    self._y_format_at_least_two_decimal_places(axis)
                    axis.set_ylabel("Cumulative return")

                if isinstance(performance.cum_returns, pd.Series):
                    performance.cum_returns.name = "strategy"
                benchmark_cum_returns = performance.benchmark_cum_returns
                try:
                    vs_benchmark = pd.concat((performance.cum_returns, benchmark_cum_returns), axis=1, sort=True)
                except TypeError:
                    # sort was introduced in pandas 0.23
                    vs_benchmark = pd.concat((performance.cum_returns, benchmark_cum_returns), axis=1)
                plot = with_baseline(vs_benchmark).plot(ax=axis, title="Cumulative Returns vs Benchmark {0}".format(extra_label))
                axis.set_xlabel("")
                if isinstance(vs_benchmark, pd.DataFrame):
                    self._clear_legend(plot, legend_title=legend_title)
