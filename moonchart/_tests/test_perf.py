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

# To run: pytest path/to/moonchart/_tests -v

import os
import unittest
import pandas as pd
# Specify non-interactive matplotlib backend before anything else imports
# matplotlib
import matplotlib as mpl
mpl.use("Agg")
from moonchart import DailyPerformance, AggregateDailyPerformance
from moonchart.utils import get_zscores
from copy import deepcopy

BACKTEST_RESULTS = {
    'strategy-1': {
        ('AbsExposure', '2018-12-03'): 0.333333,
        ('AbsExposure', '2018-12-04'): 0.333333,
        ('AbsExposure', '2018-12-05'): 0.333333,
        ('AbsWeight', '2018-12-03'): 0.333333,
        ('AbsWeight', '2018-12-04'): 0.333333,
        ('AbsWeight', '2018-12-05'): 0.333333,
        ('Commission', '2018-12-03'): 0.0001,
        ('Commission', '2018-12-04'): 0.0001,
        ('Commission', '2018-12-05'): 0.0001,
        ('NetExposure', '2018-12-03'): -0.03030303,
        ('NetExposure', '2018-12-04'): 0.060606060999999996,
        ('NetExposure', '2018-12-05'): -0.090909091,
        ('Return', '2018-12-03'): -0.002257125,
        ('Return', '2018-12-04'): -0.000375271,
        ('Return', '2018-12-05'): -0.002395708,
        ('Slippage', '2018-12-03'): 0.0001,
        ('Slippage', '2018-12-04'): 0.0001,
        ('Slippage', '2018-12-05'): 0.0001,
        ('TotalHoldings', '2018-12-03'): 22.0,
        ('TotalHoldings', '2018-12-04'): 22.0,
        ('TotalHoldings', '2018-12-05'): 22.0,
        ('Turnover', '2018-12-03'): 0.049062049,
        ('Turnover', '2018-12-04'): 0.090909091,
        ('Turnover', '2018-12-05'): 0.151515152},
    'strategy-2': {
        ('AbsExposure', '2018-12-03'): 0.333333,
        ('AbsExposure', '2018-12-04'): 0.333333,
        ('AbsExposure', '2018-12-05'): 0.333333,
        ('AbsWeight', '2018-12-03'): 0.333333,
        ('AbsWeight', '2018-12-04'): 0.333333,
        ('AbsWeight', '2018-12-05'): 0.333333,
        ('Commission', '2018-12-03'): 0.0,
        ('Commission', '2018-12-04'): 0.0,
        ('Commission', '2018-12-05'): 0.0,
        ('NetExposure', '2018-12-03'): 0.333333,
        ('NetExposure', '2018-12-04'): 0.333333,
        ('NetExposure', '2018-12-05'): 0.333333,
        ('Return', '2018-12-03'): 0.00278717,
        ('Return', '2018-12-04'): -0.005031677,
        ('Return', '2018-12-05'): -0.004845368,
        ('Slippage', '2018-12-03'): 0.0,
        ('Slippage', '2018-12-04'): 0.0,
        ('Slippage', '2018-12-05'): 0.003,
        ('TotalHoldings', '2018-12-03'): 25.0,
        ('TotalHoldings', '2018-12-04'): 25.0,
        ('TotalHoldings', '2018-12-05'): 25.0,
        ('Turnover', '2018-12-03'): 3.47e-18,
        ('Turnover', '2018-12-04'): 0.0,
        ('Turnover', '2018-12-05'): 0.0}}

PNL_RESULTS = {
    'strategy-a': {
        ('AbsExposure', '2019-01-21', '09:30:00'): '0',
        ('AbsExposure', '2019-01-21', '09:31:00'): '0.01',
        ('AbsExposure', '2019-01-22', '16:01:00'): '0',
        ('AbsExposure', '2019-01-23', '23:59:59'): '0',
        ('Account', '2019-01-21', '09:30:00'): 'U12345',
        ('Account', '2019-01-21', '09:31:00'): 'U12345',
        ('Account', '2019-01-22', '16:01:00'): 'U12345',
        ('Account', '2019-01-23', '23:59:59'): 'U12345',
        ('Commission', '2019-01-21', '09:30:00'): '0',
        ('Commission', '2019-01-21', '09:31:00'): '0',
        ('Commission', '2019-01-22', '16:01:00'): '2.54E-05',
        ('Commission', '2019-01-23', '23:59:59'): '7.11E-05',
        ('CommissionAmount', '2019-01-21', '09:30:00'): '0',
        ('CommissionAmount', '2019-01-21', '09:31:00'): '0',
        ('CommissionAmount', '2019-01-22', '16:01:00'): '15.3382',
        ('CommissionAmount', '2019-01-23', '23:59:59'): '43.691',
        ('NetExposure', '2019-01-21', '09:30:00'): '0',
        ('NetExposure', '2019-01-21', '09:31:00'): '-0.01',
        ('NetExposure', '2019-01-22', '16:01:00'): '0',
        ('NetExposure', '2019-01-23', '23:59:59'): '0',
        ('NetLiquidation', '2019-01-21', '09:30:00'): '604431.98',
        ('NetLiquidation', '2019-01-21', '09:31:00'): '604431.98',
        ('NetLiquidation', '2019-01-22', '16:01:00'): '604346.46',
        ('NetLiquidation', '2019-01-23', '23:59:59'): '614640.04',
        ('OrderRef', '2019-01-21', '09:30:00'): 'strategy-a',
        ('OrderRef', '2019-01-21', '09:31:00'): 'strategy-a',
        ('OrderRef', '2019-01-22', '16:01:00'): 'strategy-a',
        ('OrderRef', '2019-01-23', '23:59:59'): 'strategy-a',
        ('Pnl', '2019-01-21', '09:30:00'): '0',
        ('Pnl', '2019-01-21', '09:31:00'): '0',
        ('Pnl', '2019-01-22', '16:01:00'): '732.6318',
        ('Pnl', '2019-01-23', '23:59:59'): '2463.289',
        ('Return', '2019-01-21', '09:30:00'): '0',
        ('Return', '2019-01-21', '09:31:00'): '0',
        ('Return', '2019-01-22', '16:01:00'): '0.00121226',
        ('Return', '2019-01-23', '23:59:59'): '0.00400769',
        ('TotalHoldings', '2019-01-21', '09:30:00'): 25.0,
        ('TotalHoldings', '2019-01-21', '09:31:00'): 26,
        ('TotalHoldings', '2019-01-22', '16:01:0'): 25.0,
        ('TotalHoldings', '2019-01-23', '23:59:59'): 25.0,
        ('Turnover', '2019-01-21', '09:30:00'): 3.47e-18,
        ('Turnover', '2019-01-21', '09:31:00'): 0,
        ('Turnover', '2019-01-22', '16:01:0'): 0.0,
        ('Turnover', '2019-01-23', '23:59:59'): 0.0},
    'strategy-b': {
        ('AbsExposure', '2019-01-21', '09:30:00'): '0',
        ('AbsExposure', '2019-01-22', '16:01:00'): '0',
        ('AbsExposure', '2019-01-23', '23:59:59'): '0',
        ('Account', '2019-01-21', '09:30:00'): 'U12345',
        ('Account', '2019-01-22', '16:01:00'): 'U12345',
        ('Account', '2019-01-23', '23:59:59'): 'U12345',
        ('Commission', '2019-01-21', '09:30:00'): '0',
        ('Commission', '2019-01-22', '16:01:00'): '0.000179206',
        ('Commission', '2019-01-23', '23:59:59'): '5.55E-05',
        ('CommissionAmount', '2019-01-21', '09:30:00'): '0',
        ('CommissionAmount', '2019-01-22', '16:01:00'): '108.3024',
        ('CommissionAmount', '2019-01-23', '23:59:59'): '34.0915',
        ('NetExposure', '2019-01-21', '09:30:00'): '0',
        ('NetExposure', '2019-01-22', '16:01:00'): '0',
        ('NetExposure', '2019-01-23', '23:59:59'): '0',
        ('NetLiquidation', '2019-01-21', '09:30:00'): '604431.98',
        ('NetLiquidation', '2019-01-22', '16:01:00'): '604346.46',
        ('NetLiquidation', '2019-01-23', '23:59:59'): '614640.04',
        ('OrderRef', '2019-01-21', '09:30:00'): 'strategy-b',
        ('OrderRef', '2019-01-22', '16:01:00'): 'strategy-b',
        ('OrderRef', '2019-01-23', '23:59:59'): 'strategy-b',
        ('Pnl', '2019-01-21', '09:30:00'): '0',
        ('Pnl', '2019-01-22', '16:01:00'): '501.1911',
        ('Pnl', '2019-01-23', '23:59:59'): '6534.1285',
        ('Return', '2019-01-21', '09:30:00'): '0',
        ('Return', '2019-01-22', '16:01:00'): '0.0008293',
        ('Return', '2019-01-23', '23:59:59'): '0.01063083',
        ('TotalHoldings', '2019-01-21', '09:30:00'): 15.0,
        ('TotalHoldings', '2019-01-22', '16:01:00'): 20.0,
        ('TotalHoldings', '2019-01-23', '23:59:59'): 22.0,
        ('Turnover', '2019-01-21', '09:30:00'): 3.47e-18,
        ('Turnover', '2019-01-22', '16:01:00'): 0.01,
        ('Turnover', '2019-01-23', '23:59:59'): 0.02}}

def round_results(results_dict_or_list, n=6):
    """
    Rounds the values in results_dict, which can be scalars or
    lists.
    """
    if isinstance(results_dict_or_list, dict):
        for key, value in results_dict_or_list.items():
            if isinstance(value, list):
                results_dict_or_list[key] = [round(v, n) for v in value]
            else:
                results_dict_or_list[key] = round(value, n)
        return results_dict_or_list
    else:
        return [round(value, n) for value in results_dict_or_list]

class DailyPerformanceTestCase(unittest.TestCase):
    """
    Test cases for DailyPerformance and AggregateDailyPerformance.
    """

    def setUp(self):
        """
        Write test fixtures to CSV.
        """
        backtest_results = pd.DataFrame.from_dict(BACKTEST_RESULTS)
        backtest_results.index.set_names(["Field","Date"], inplace=True)
        backtest_results.to_csv("backtest.csv")

        pnl_results = pd.DataFrame.from_dict(PNL_RESULTS)
        pnl_results.index.set_names(["Field","Date", "Time"], inplace=True)
        pnl_results.to_csv("pnl.csv")

    def tearDown(self):
        """
        Remove files.
        """
        os.remove("backtest.csv")
        os.remove("pnl.csv")

    def test_from_moonshot_csv(self):

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")
        self.assertListEqual(
            list(perf.returns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05']
        )
        self.assertDictEqual(
            round_results(perf.returns.to_dict(orient="list")),
            {'strategy-1': [-0.002257, -0.000375, -0.002396],
             'strategy-2': [0.002787, -0.005032, -0.004845]})

        self.assertListEqual(
            list(perf.abs_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            round_results(perf.abs_exposures.to_dict(orient="list")),
            {'strategy-1': [0.333333, 0.333333, 0.333333],
             'strategy-2': [0.333333, 0.333333, 0.333333]})

        self.assertListEqual(
            list(perf.net_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            round_results(perf.net_exposures.to_dict(orient="list")),
            {'strategy-1': [-0.030303, 0.060606, -0.090909],
             'strategy-2': [0.333333, 0.333333, 0.333333]})

        self.assertListEqual(
            list(perf.turnover.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            round_results(perf.turnover.to_dict(orient="list")),
            {'strategy-1': [0.049062, 0.090909, 0.151515],
             'strategy-2': [0.0, 0.0, 0.0]})

        self.assertListEqual(
            list(perf.total_holdings.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.total_holdings.to_dict(orient="list"),
            {'strategy-1': [22.0, 22.0, 22.0], 'strategy-2': [25.0, 25.0, 25.0]})

        self.assertListEqual(
            list(perf.commissions.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.commissions.to_dict(orient="list"),
            {'strategy-1': [0.0001, 0.0001, 0.0001], 'strategy-2': [0.0, 0.0, 0.0]})

        self.assertListEqual(
            list(perf.cum_commissions.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.cum_commissions.to_dict(orient="list"),
            {'strategy-1': [1.0001, 1.0002, 1.0003], 'strategy-2': [1.0, 1.0, 1.0]})

        self.assertListEqual(
            list(perf.slippages.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.slippages.to_dict(orient="list"),
            {'strategy-1': [0.0001, 0.0001, 0.0001], 'strategy-2': [0.0, 0.0, 0.003]})

        self.assertListEqual(
            list(perf.cum_slippages.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.cum_slippages.to_dict(orient="list"),
            {'strategy-1': [1.0001, 1.0002, 1.0003], 'strategy-2': [1.0, 1.0, 1.003]})

        self.assertDictEqual(
            round_results(perf.cagr.to_dict()),
            {'strategy-1': -0.600935, 'strategy-2': -0.727217})

        self.assertDictEqual(
            round_results(perf.sharpe.to_dict()),
            {'strategy-1': -23.57405, 'strategy-2': -8.409034}
        )

        self.assertListEqual(
            list(perf.cum_returns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            round_results(perf.cum_returns.to_dict(orient="list")),
            {'strategy-1': [0.997743, 0.997368, 0.994979],
            'strategy-2': [1.002787, 0.997741, 0.992907]}
        )

        self.assertListEqual(
            list(perf.drawdowns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            round_results(perf.drawdowns.to_dict(orient="list")),
            {'strategy-1': [0.0, -0.000375, -0.00277],
             'strategy-2': [0.0, -0.005032, -0.009853]}
        )

        self.assertDictEqual(
            round_results(perf.max_drawdown.to_dict()),
            {'strategy-1': -0.00277, 'strategy-2': -0.009853})

    def test_from_moonshot_csv_agg_perf(self):

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")
        agg_perf = AggregateDailyPerformance(perf)

        self.assertListEqual(
            list(agg_perf.returns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05']
        )
        self.assertListEqual(
            round_results(agg_perf.returns.tolist()),
            [0.00053, -0.005407, -0.007241])

        self.assertListEqual(
            list(agg_perf.abs_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            round_results(agg_perf.abs_exposures.tolist()),
            [0.666666, 0.666666, 0.666666])

        self.assertListEqual(
            list(agg_perf.net_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            round_results(agg_perf.net_exposures.tolist()),
            [0.30303, 0.393939, 0.242424])

        self.assertListEqual(
            list(agg_perf.turnover.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            round_results(agg_perf.turnover.tolist()),
            [0.049062, 0.090909, 0.151515])

        self.assertListEqual(
            list(agg_perf.total_holdings.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.total_holdings.tolist(),
            [47.0, 47.0, 47.0])

        self.assertListEqual(
            list(agg_perf.commissions.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.commissions.tolist(),
            [0.0001, 0.0001, 0.0001])

        self.assertListEqual(
            list(agg_perf.cum_commissions.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.cum_commissions.tolist(),
            [1.0001, 1.0002, 1.0003])

        self.assertListEqual(
            list(agg_perf.slippages.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.slippages.tolist(),
            [0.0001, 0.0001, 0.0031])

        self.assertListEqual(
            list(agg_perf.cum_slippages.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.cum_slippages.tolist(),
            [1.0001, 1.0002, 1.0033])

        self.assertEqual(
            round(agg_perf.cagr, 6),-0.891287)

        self.assertEqual(
            round(agg_perf.sharpe, 6), -15.785645)

        self.assertListEqual(
            list(agg_perf.cum_returns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            round_results(agg_perf.cum_returns.tolist()),
            [1.00053, 0.99512, 0.987914])

        self.assertListEqual(
            list(agg_perf.drawdowns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            round_results(agg_perf.drawdowns.tolist()),
            [0.0, -0.005407, -0.012609]
        )

        self.assertEqual(
            round(agg_perf.max_drawdown, 6), -0.012609)

    def test_from_pnl_csv(self):

        self.maxDiff = None

        perf = DailyPerformance.from_pnl_csv("pnl.csv")
        self.assertListEqual(
            list(perf.returns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00']
        )
        self.assertDictEqual(
            round_results(perf.returns.to_dict(orient="list")),
            {'strategy-a': [0.0, 0.001212, 0.004008],
            'strategy-b': [0.0, 0.000829, 0.010631]})

        self.assertListEqual(
            list(perf.abs_exposures.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.abs_exposures.to_dict(orient="list"),
            {'strategy-a': [0.01, 0.0, 0.0], 'strategy-b': [0.0, 0.0, 0.0]})

        self.assertListEqual(
            list(perf.net_exposures.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.net_exposures.to_dict(orient="list"),
            {'strategy-a': [-0.01, 0.0, 0.0], 'strategy-b': [0.0, 0.0, 0.0]})

        self.assertListEqual(
            list(perf.turnover.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            round_results(perf.turnover.to_dict(orient="list")),
            {'strategy-a': [0.0, 0.0, 0.0], 'strategy-b': [0.0, 0.01, 0.02]})

        self.assertListEqual(
            list(perf.total_holdings.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.total_holdings.to_dict(orient="list"),
            {'strategy-a': [26.0, 25.0, 25.0], 'strategy-b': [15.0, 20.0, 22.0]})

        self.assertListEqual(
            list(perf.pnl.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.pnl.to_dict(orient="list"),
            {'strategy-a': [0.0, 732.6318, 2463.289],
             'strategy-b': [0.0, 501.1911, 6534.1285]})

        self.assertListEqual(
            list(perf.commission_amounts.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.commission_amounts.to_dict(orient="list"),
            {'strategy-a': [0.0, 15.3382, 43.691], 'strategy-b': [0.0, 108.3024, 34.0915]})

        self.assertListEqual(
            list(perf.cum_commission_amounts.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.cum_commission_amounts.to_dict(orient="list"),
            {'strategy-a': [0.0, 15.3382, 59.0292], 'strategy-b': [0.0, 108.3024, 142.3939]})

        self.assertListEqual(
            list(perf.cum_pnl.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            round_results(perf.cum_pnl.to_dict(orient="list")),
           {'strategy-a': [0.0, 732.6318, 3195.9208],
            'strategy-b': [0.0, 501.1911, 7035.3196]})

        self.assertListEqual(
            list(perf.commissions.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            round_results(perf.commissions.to_dict(orient="list")),
            {'strategy-a': [0.0, 2.5e-05, 7.1e-05],
            'strategy-b': [0.0, 0.000179, 5.6e-05]})

        self.assertDictEqual(
            round_results(perf.cagr.to_dict()),
            {'strategy-a': 1.588414, 'strategy-b': 7.013847})

        self.assertDictEqual(
            round_results(perf.sharpe.to_dict()),
            {'strategy-a': 13.43909, 'strategy-b': 10.255814}
        )

        self.assertListEqual(
            list(perf.cum_returns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            round_results(perf.cum_returns.to_dict(orient="list")),
            {'strategy-a': [1.0, 1.001212, 1.005225],
            'strategy-b': [1.0, 1.000829, 1.011469]}
        )

        self.assertListEqual(
            list(perf.drawdowns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.drawdowns.to_dict(orient="list"),
            {'strategy-a': [0.0, 0.0, 0.0], 'strategy-b': [0.0, 0.0, 0.0]})

        self.assertDictEqual(
            perf.max_drawdown.to_dict(),
            {'strategy-a': 0.0, 'strategy-b': 0.0})

    def test_from_pnl_csv_agg_perf(self):

        perf = DailyPerformance.from_pnl_csv("pnl.csv")
        agg_perf = AggregateDailyPerformance(perf)

        self.assertListEqual(
            list(agg_perf.returns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00']
        )
        self.assertListEqual(
            round_results(agg_perf.returns.tolist()),
            [0.0, 0.002042, 0.014639])

        self.assertListEqual(
            list(agg_perf.abs_exposures.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.abs_exposures.tolist(),
            [0.01, 0.0, 0.0])

        self.assertListEqual(
            list(agg_perf.net_exposures.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.net_exposures.tolist(),
            [-0.01, 0.0, 0.0])

        self.assertListEqual(
            list(agg_perf.turnover.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            round_results(agg_perf.turnover.tolist()),
            [0.0, 0.01, 0.02])

        self.assertListEqual(
            list(agg_perf.total_holdings.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.total_holdings.tolist(),
            [41.0, 45.0, 47.0])

        self.assertListEqual(
            list(agg_perf.pnl.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            round_results(agg_perf.pnl.tolist()),
            [0.0, 1233.8229, 8997.4175])

        self.assertListEqual(
            list(agg_perf.commission_amounts.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.commission_amounts.tolist(),
            [0.0, 123.6406, 77.7825])

        self.assertListEqual(
            list(agg_perf.cum_commission_amounts.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.cum_commission_amounts.tolist(),
            [0.0, 123.6406, 201.4231])

        self.assertListEqual(
            list(agg_perf.cum_pnl.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            round_results(agg_perf.cum_pnl.tolist()),
           [0.0, 1233.8229, 10231.2404])

        self.assertListEqual(
            list(agg_perf.commissions.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            round_results(agg_perf.commissions.tolist()),
            [0.0, 0.000205, 0.000127])

        self.assertEqual(
            round(agg_perf.cagr, 6), 19.581033)

        self.assertEqual(
            round(agg_perf.sharpe, 6), 11.13276)

        self.assertListEqual(
            list(agg_perf.cum_returns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            round_results(agg_perf.cum_returns.tolist()),
            [1.0, 1.002042, 1.01671]
        )

        self.assertListEqual(
            list(agg_perf.drawdowns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.drawdowns.tolist(),
            [0.0, 0.0, 0.0])

        self.assertEqual(
            agg_perf.max_drawdown, 0.0)

    def test_override_how_to_aggregate(self):

        self.maxDiff = None

        perf = DailyPerformance.from_pnl_csv("pnl.csv", how_to_aggregate={"TotalHoldings":"mean"})
        self.assertListEqual(
            list(perf.returns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00']
        )
        self.assertDictEqual(
            perf.returns.to_dict(orient="list"),
            {'strategy-a': [0.0, 0.00121226, 0.00400769],
             'strategy-b': [0.0, 0.0008293, 0.01063083]})

        self.assertListEqual(
            list(perf.abs_exposures.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.abs_exposures.to_dict(orient="list"),
            {'strategy-a': [0.01, 0.0, 0.0], 'strategy-b': [0.0, 0.0, 0.0]})

        self.assertListEqual(
            list(perf.net_exposures.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.net_exposures.to_dict(orient="list"),
            {'strategy-a': [-0.01, 0.0, 0.0], 'strategy-b': [0.0, 0.0, 0.0]})

        self.assertListEqual(
            list(perf.turnover.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.turnover.to_dict(orient="list"),
            {'strategy-a': [3.47e-18, 0.0, 0.0], 'strategy-b': [3.47e-18, 0.01, 0.02]})

        self.assertListEqual(
            list(perf.total_holdings.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.total_holdings.to_dict(orient="list"),
            {'strategy-a': [25.5, 25.0, 25.0], 'strategy-b': [15.0, 20.0, 22.0]})

    def test_riskfree(self):

        perf = DailyPerformance.from_pnl_csv("pnl.csv")
        self.assertDictEqual(
            round_results(perf.sharpe.to_dict()),
            {'strategy-a': 13.43909, 'strategy-b': 10.255814}
        )

        perf = DailyPerformance.from_pnl_csv("pnl.csv", riskfree=0.02/252)
        self.assertDictEqual(
            round_results(perf.sharpe.to_dict()),
            {'strategy-a': 12.826098, 'strategy-b': 10.04274}
        )

    def test_compound(self):

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")
        self.assertDictEqual(
            round_results(perf.cum_returns.to_dict(orient="list")),
            {'strategy-1': [0.997743, 0.997368, 0.994979],
            'strategy-2': [1.002787, 0.997741, 0.992907]}
        )

        perf = DailyPerformance.from_moonshot_csv("backtest.csv", compound=False)
        self.assertDictEqual(
            round_results(perf.cum_returns.to_dict(orient="list")),
            {'strategy-1': [0.997743, 0.997368, 0.994972],
            'strategy-2': [1.002787, 0.997755, 0.99291]}
        )

    def test_rolling_sharpe_window(self):

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")
        self.assertDictEqual(
            perf.rolling_sharpe.fillna(-1).to_dict(orient="list"),
            {'strategy-1': [-1.0, -1.0, -1.0],
             'strategy-2': [-1.0, -1.0, -1.0]}
        )

        perf = DailyPerformance.from_moonshot_csv("backtest.csv", rolling_sharpe_window=2)
        self.assertDictEqual(
            perf.rolling_sharpe.fillna(-1).to_dict(orient="list"),
            {'strategy-1': [-1.0, -22.205756137004837, -21.77149197579271],
             'strategy-2': [-1.0, -4.55699466016689, -841.576244567701]}
        )

    def test_trim_outliers(self):

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")
        zscores = get_zscores(perf.returns)

        self.assertDictEqual(
            round_results(zscores.to_dict(orient="list")),
            {'strategy-1': [-0.514866, 1.152522, -0.637656],
             'strategy-2': [1.154449, -0.598104, -0.556344]}
        )

        self.assertDictEqual(
           round_results( perf.returns.to_dict(orient="list")),
            {'strategy-1': [-0.002257, -0.000375, -0.002396],
            'strategy-2': [0.002787, -0.005032, -0.004845]}
        )

        perf = DailyPerformance.from_moonshot_csv("backtest.csv", trim_outliers=1.154)
        self.assertDictEqual(
            round_results(perf.returns.to_dict(orient="list")),
            {'strategy-1': [-0.002257, -0.000375, -0.002396],
            'strategy-2': [0.0, -0.005032, -0.004845]}
        )

    def test_benchmark(self):

        backtest_results = deepcopy(BACKTEST_RESULTS)
        backtest_results["strategy-1"].update(
            {('Benchmark', '2018-12-03'): None,
            ('Benchmark', '2018-12-04'): None,
            ('Benchmark', '2018-12-05'): None}
        )
        backtest_results["strategy-2"].update(
            {('Benchmark', '2018-12-03'): 100.10,
            ('Benchmark', '2018-12-04'): 102.34,
            ('Benchmark', '2018-12-05'): 102.08}
        )

        backtest_results = pd.DataFrame.from_dict(backtest_results)
        backtest_results.index.set_names(["Field","Date"], inplace=True)
        backtest_results.to_csv("backtest.csv")

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")

        self.assertListEqual(
            round_results(perf.benchmark_returns.tolist()),
            [0.0, 0.022378, -0.002541]
        )