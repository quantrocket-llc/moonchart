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

class ShortFallTearsheet(BaseTearsheet):

    def create_shortfall_tearsheet(self, live_returns, simulated_returns):
        """
        Shows drawdown and equity curves comparing the live and simulated
        returns. DataFrames should include a column for each strategy being
        analyzed.
        """
        raise NotImplementedError()

        # TODO: be more specific
        self.suptitle = "Shortfall Analysis"

        strategies = live_returns.columns.union(simulated_returns.columns)
        strategy_count = len(strategies)

        rows, cols = self._get_plot_dimensions(strategy_count)

        for i, strategy in enumerate(strategies):
            if strategy in live_returns.columns:
                strategy_live_returns = live_returns[strategy]
            else:
                strategy_live_returns = pd.Series(0, index=live_returns.index)
            strategy_live_returns.name = "live"
            strategy_simulated_returns = simulated_returns[strategy]
            strategy_simulated_returns.name = "simulated"
            try:
                strategy_returns = pd.concat((strategy_live_returns, strategy_simulated_returns), axis=1, sort=True).fillna(0)
            except TypeError:
                # sort was introduced in pandas 0.23
                strategy_returns = pd.concat((strategy_live_returns, strategy_simulated_returns), axis=1).fillna(0)
            cum_returns = self.cum_returns(self._with_baseline(strategy_returns))
            drawdowns = self.drawdowns(cum_returns)
            shortfall = cum_returns.live - cum_returns.simulated

            fig = plt.figure("Cumulative Returns", figsize=self.window_size)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(rows, cols, i+1)
            cum_returns.plot(ax=axis, title=strategy)

            fig = plt.figure("Drawdowns", figsize=self.window_size)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(rows, cols, i+1)
            drawdowns.plot(ax=axis, title=strategy)

            fig = plt.figure("Shortfall", figsize=self.window_size)
            fig.suptitle(self.suptitle, **self.suptitle_kwargs)
            axis = fig.add_subplot(rows, cols, i+1)
            shortfall.plot(ax=axis, title=strategy, kind="area", stacked=False)

        self._save_or_show()