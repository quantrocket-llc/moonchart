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

import pandas as pd
import matplotlib.pyplot as plt
from .base import BaseTearsheet

class ParamscanTearsheet(BaseTearsheet):

    def create_paramscan_tearsheet(self, results):

        # Parse csv
        results = pd.read_csv("paramscan_results.csv")
        idx_cols=list(results.columns)
        idx_cols.remove("Value")
        results = results.set_index(idx_cols)
        results = results["Value"] # 1-col DataFrame to Series
        idx_cols.remove("Field")
        idx_cols.remove("StrategyOrDate")
        params = idx_cols
        results = results.unstack(level=params)

        # if 1d, create bar plots for CAGR, Sharpe, Max DD etc
        sharpes = results.loc["Sharpe"]
        sharpes.index.name = "Strategy"
        sharpes.T.plot(kind="bar")

        # if 2D, create a sheet with a plot for each strategy, and do a heatmap
        fig = plt.figure("MAVG_WINDOW + FOO")
        # see shortfall command for subplot calculations
        ax = fig.add_subplot(111, title="etf-demo")
        sns.heatmap(sharpes.T["etf-demo"].unstack(), annot=True,
                annot_kws={"size": 9},
                center=0.0,
                cbar=False,
                ax=ax,
                cmap=matplotlib.cm.RdYlGn)

        # Create performance from agg_returns
        # Plot cum returns, max dd for details but not agg

        cagrs.name = "CAGR"
        sharpes.name = "Sharpe"
        max_drawdowns.name = "Max Drawdown"
        abs_exposures.name = "Absolute Exposure"
        normalized_cagrs.name = "Normalized CAGR (CAGR/Exposure)"

        series = [cagrs, sharpes, max_drawdowns, abs_exposures, normalized_cagrs]

        title = param1
        if param2:
            title = "{0} and {1}".format(title, param2)
        if date_range_msg:
            title = "{0} ({1})".format(title, date_range_msg)

        performance = agg_performance = Performance(agg_returns)

        tearsheet = Tearsheet(pdf_filename=outfile)
        tearsheet.suptitle = title
        tearsheet.plot_arbitrary(series, kind="bar", title=title)
        tearsheet.create_performance_tearsheet(performance, agg_performance)

    def plot_arbitrary(self, series, kind="line", title=None):
        """
        Plots an arbitrary list of Series or DataFrames. The Series or
        DataFrame names are used as the chart titles.
        """
        series_list = series
        rows, cols = self._get_plot_dimensions(len(series))
        fig = plt.figure(figsize=self.window_size)
        if title:
            self.suptitle = title

        fig.suptitle(self.suptitle, **self.suptitle_kwargs)
        for i, series in enumerate(series_list):
            axis = fig.add_subplot(rows, cols, i + 1)
            plot = series.plot(ax=axis, kind=kind, title=series.name, fontsize="small")
            if isinstance(series, pd.DataFrame):
                self._clear_legend(plot)

        self._save_or_show()

