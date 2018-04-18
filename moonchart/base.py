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
from matplotlib import cycler
from matplotlib.backends.backend_pdf import PdfPages

class BaseTearsheet(object):
    """
    Base class for tear sheets.
    """
    DEFAULT_TITLE = "Performance tear sheet"

    def __init__(self, pdf_filename=None, window_size=None, max_cols_for_details=25):
        self.window_size = window_size or (12.0, 7.5) # width, height in inches
        plt.rc("legend", fontsize="xx-small")
        plt.rc("axes",
               prop_cycle=cycler("color", [
                   "b", "g", "r", "c", "m", "y", "k",
                   "sienna", "chartreuse", "darkorange", "springgreen", "gray",
                   "powderblue", "cornflowerblue", "maroon", "indigo", "deeppink",
                   "salmon", "darkseagreen", "rosybrown", "slateblue", "darkgoldenrod",
                   "deepskyblue",
               ]),
               facecolor="#e1e1e6",
               edgecolor="#aaaaaa",
               grid=True,
               axisbelow=True)
        plt.rc("grid", linestyle="-", color="#ffffff")
        plt.rc("figure", autolayout=True)
        plt.rc("xtick", labelsize="xx-small")
        plt.rc("ytick", labelsize="xx-small")
        if pdf_filename:
            self.pdf = PdfPages(pdf_filename, keep_empty=True)
        else:
            self.pdf = None

        self.suptitle = self.DEFAULT_TITLE
        self.suptitle_kwargs = {
            "bbox": dict(facecolor="#e1e1e6", edgecolor='#aaaaaa', alpha=0.5)}
        self.max_cols_for_details = max_cols_for_details

    def _save_or_show(self):
        """
        Saves the fig to the multi-page PDF, or shows it.
        """
        if self.pdf:
            for fignum in plt.get_fignums():
                self.pdf.savefig(fignum)
            plt.close("all")
            self.pdf.close()
        else:
            plt.show()

    def _get_plot_dimensions(self, plot_count):
        """
        Returns a tuple of rows, cols needed to accomodate the plot_count.
        """
        rows = math.ceil(math.sqrt(plot_count))
        cols = math.ceil(plot_count/rows)
        return rows, cols

    @property
    def _tight_layout_clear_suptitle(self):
        # leave room at top for suptitle
        return dict(rect=[0,0,1,.9])

    def _clear_legend(self, plot, legend_title=None):
        """
        Anchors the legend to the outside right of the plot area so you can
        see the plot.
        """
        plot.figure.set_tight_layout({"pad":10, "h_pad":1, "w_pad":1})
        plot.legend(
            loc='center left', bbox_to_anchor=(1, 0.5), fontsize="x-small", title=legend_title)

    def _create_performance_plots(self, performance, subplot, extra_label):
        """
        Creates agg/details plots for cumulative returns, drawdowns, rolling
        Sharpe, and possibly pnl.
        """
        if subplot == 111:
            tight_layout = self._tight_layout_clear_suptitle
        else:
            tight_layout = None

        fig = plt.figure("Cumulative Returns", figsize=self.window_size, tight_layout=tight_layout)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(subplot)
        max_return = performance.cum_returns.max(axis=0)
        if isinstance(max_return, pd.Series):
            max_return = max_return.max(axis=0)
        # If the price more than doubled, use a log scale
        if max_return >= 2:
            axis.set_yscale("log", basey=2)

        include_commissions = (
            performance.commissions_pct is not None
            # if all commissions are null/0, don't show them
            and (performance.commissions_pct.fillna(0) != 0).any())

        include_slippage = (
            performance.slippages is not None
            # if all slippages are null/0, don't show them
            and (performance.slippages.fillna(0) != 0).any())

        if (
            # a 212 subplot means a detailed plot, which isn't compatible with
            # showing commissions and slippage
            subplot != 212 and (include_commissions or include_slippage)):

            if include_commissions:
                commissions_pct = performance.with_baseline(performance.commissions_pct)
                cum_commissions_pct = performance.get_cum_returns(commissions_pct)
                cum_commissions_pct.name = "commissions"

            if include_slippage:
                slippages = performance.with_baseline(performance.slippages)
                cum_slippages = performance.get_cum_returns(slippages)
                cum_slippages.name = "slippage"

            performance.cum_returns_with_baseline.name = "returns"

            cum_gross_returns = performance.cum_returns_with_baseline

            if include_commissions:
                cum_gross_returns = cum_gross_returns.multiply(cum_commissions_pct)

            if include_slippage:
                cum_gross_returns = cum_gross_returns.multiply(cum_slippages)

            cum_gross_returns.name = "gross returns"
            breakdown_parts = [performance.cum_returns_with_baseline, cum_gross_returns]

            if include_commissions:
                breakdown_parts.append(cum_commissions_pct)

            if include_slippage:
                breakdown_parts.append(cum_slippages)

            returns_breakdown = pd.concat(breakdown_parts, axis=1)
            plot = returns_breakdown.plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
            if isinstance(returns_breakdown, pd.DataFrame):
                self._clear_legend(plot)
        else:
            plot = performance.cum_returns_with_baseline.plot(ax=axis, title="Cumulative Returns {0}".format(extra_label))
            if isinstance(performance.cum_returns_with_baseline, pd.DataFrame):
                self._clear_legend(plot)

        fig = plt.figure("Drawdowns", figsize=self.window_size, tight_layout=tight_layout)
        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        axis = fig.add_subplot(subplot)
        plot = performance.drawdowns.plot(ax=axis, title="Drawdowns {0}".format(extra_label))
        if isinstance(performance.drawdowns, pd.DataFrame):
            self._clear_legend(plot)

        if performance.cum_pnl is not None:
            # Cumulative pnl will generally be very similar to cumulative
            # return, but they might be different if cumulative return is
            # calcuated based on an account balance that changes due to
            # causes other than the strategies being charted (e.g. other
            # strategies, contributions/withdrawals, etc.)
            fig = plt.figure("Cumulative PNL", figsize=self.window_size,
                             tight_layout=self._tight_layout_clear_suptitle)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(subplot)
            if (
                performance.commissions is not None
                # a 212 subplot means a detailed plot, which isn't compatible with
                # showing commissions
                and subplot != 212
                # if all commissions are null/0, don't show them
                and (performance.commissions.fillna(0) != 0).any()):
                cum_commissions = performance.commissions.cumsum()
                cum_commissions.name = "commissions"
                cum_pnl = performance.pnl.cumsum()
                cum_pnl.name = "pnl"
                cum_gross_pnl = cum_pnl + cum_commissions.abs()
                cum_gross_pnl.name = "gross pnl"
                pnl_breakdown = pd.concat((cum_pnl, cum_gross_pnl, cum_commissions), axis=1)
                plot = pnl_breakdown.plot(ax=axis, title="Cumulative PNL {0}".format(extra_label))
                if isinstance(returns_breakdown, pd.DataFrame):
                    self._clear_legend(plot)
            else:
                plot = performance.cum_pnl.plot(ax=axis, title="Cumulative PNL {0}".format(extra_label))
                if isinstance(performance.cum_pnl, pd.DataFrame):
                    self._clear_legend(plot)

        if len(performance.rolling_sharpe.index) > performance.rolling_sharpe_window:
            fig = plt.figure("Rolling Sharpe", figsize=self.window_size, tight_layout=tight_layout)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(subplot)
            plot = performance.rolling_sharpe.plot(ax=axis, title="Rolling Sharpe ({0}-day) {1}".format(
                performance.rolling_sharpe_window, extra_label))
            if isinstance(performance.rolling_sharpe, pd.DataFrame):
                self._clear_legend(plot)

        benchmark_returns = performance.get_benchmark_returns()
        if benchmark_returns is not None:
            fig = plt.figure("Cumulative Returns vs Benchmark", figsize=self.window_size, tight_layout=tight_layout)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(subplot)
            max_return = performance.cum_returns.max(axis=0)
            if isinstance(max_return, pd.Series):
                max_return = max_return.max(axis=0)
            # If the price more than doubled, use a log scale
            if max_return >= 2:
                axis.set_yscale("log", basey=2)

            if isinstance(performance.cum_returns_with_baseline, pd.Series):
                performance.cum_returns_with_baseline.name = "strategy"
            benchmark_cum_returns = performance.get_cum_returns(performance.with_baseline(benchmark_returns))
            vs_benchmark = pd.concat((performance.cum_returns_with_baseline, benchmark_cum_returns), axis=1)
            plot = vs_benchmark.plot(ax=axis, title="Cumulative Returns vs Benchmark {0}".format(extra_label))
            if isinstance(vs_benchmark, pd.DataFrame):
                self._clear_legend(plot)
