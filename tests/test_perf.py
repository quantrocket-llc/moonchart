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

# To run: python3 -m unittest discover -s tests/ -p test_*.py -t . -v

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
        ('AbsExposure', '2018-12-03'): 0.333333333,
        ('AbsExposure', '2018-12-04'): 0.333333333,
        ('AbsExposure', '2018-12-05'): 0.333333333,
        ('AbsWeight', '2018-12-03'): 0.333333333,
        ('AbsWeight', '2018-12-04'): 0.333333333,
        ('AbsWeight', '2018-12-05'): 0.333333333,
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
        ('AbsExposure', '2018-12-03'): 0.333333333,
        ('AbsExposure', '2018-12-04'): 0.333333333,
        ('AbsExposure', '2018-12-05'): 0.333333333,
        ('AbsWeight', '2018-12-03'): 0.333333333,
        ('AbsWeight', '2018-12-04'): 0.333333333,
        ('AbsWeight', '2018-12-05'): 0.333333333,
        ('Commission', '2018-12-03'): 0.0,
        ('Commission', '2018-12-04'): 0.0,
        ('Commission', '2018-12-05'): 0.0,
        ('NetExposure', '2018-12-03'): 0.333333333,
        ('NetExposure', '2018-12-04'): 0.333333333,
        ('NetExposure', '2018-12-05'): 0.333333333,
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
            perf.returns.to_dict(orient="list"),
            {'strategy-1': [-0.002257125, -0.000375271, -0.002395708],
             'strategy-2': [0.00278717, -0.005031677, -0.004845368]})

        self.assertListEqual(
            list(perf.abs_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.abs_exposures.to_dict(orient="list"),
            {'strategy-1': [0.333333333, 0.333333333, 0.333333333],
             'strategy-2': [0.333333333, 0.333333333, 0.333333333]})

        self.assertListEqual(
            list(perf.net_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.net_exposures.to_dict(orient="list"),
            {'strategy-1': [-0.03030303, 0.060606061, -0.090909091],
             'strategy-2': [0.333333333, 0.333333333, 0.333333333]})

        self.assertListEqual(
            list(perf.turnover.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.turnover.to_dict(orient="list"),
            {'strategy-1': [0.049062049, 0.090909091, 0.151515152],
             'strategy-2': [3.47e-18, 0.0, 0.0]})

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
            perf.cagr.to_dict(),
            {'strategy-1': -0.6009354029100387, 'strategy-2': -0.7272165531290371})

        self.assertDictEqual(
            perf.sharpe.to_dict(),
            {'strategy-1': -23.574049805803934, 'strategy-2': -8.409034049060317}
        )

        self.assertListEqual(
            list(perf.cum_returns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.cum_returns.to_dict(orient="list"),
            {'strategy-1': [0.997742875, 0.9973684510335559, 0.9949790474564671],
             'strategy-2': [1.00278717, 0.9977414688608159, 0.9929070442753247]}
        )

        self.assertListEqual(
            list(perf.drawdowns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertDictEqual(
            perf.drawdowns.to_dict(orient="list"),
            {'strategy-1': [0.0, -0.00037527100000001035, -0.0027700799602632387],
             'strategy-2': [0.0, -0.005031676999999957, -0.009852664673277833]}
        )

        self.assertDictEqual(
            perf.max_drawdown.to_dict(),
            {'strategy-1': -0.0027700799602632387, 'strategy-2': -0.009852664673277833})

    def test_from_moonshot_csv_agg_perf(self):

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")
        agg_perf = AggregateDailyPerformance(perf)

        self.assertListEqual(
            list(agg_perf.returns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05']
        )
        self.assertListEqual(
            agg_perf.returns.tolist(),
            [0.0005300449999999998, -0.005406948, -0.007241076])

        self.assertListEqual(
            list(agg_perf.abs_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.abs_exposures.tolist(),
            [0.666666666, 0.666666666, 0.666666666])

        self.assertListEqual(
            list(agg_perf.net_exposures.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.net_exposures.tolist(),
            [0.303030303, 0.393939394, 0.242424242])

        self.assertListEqual(
            list(agg_perf.turnover.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.turnover.tolist(),
            [0.04906204900000001, 0.090909091, 0.151515152])

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
            agg_perf.cagr,-0.8912867832023363)

        self.assertEqual(
            agg_perf.sharpe, -15.785645344093775)

        self.assertListEqual(
            list(agg_perf.cum_returns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.cum_returns.tolist(),
            [1.000530045, 0.9951202310742474, 0.9879144898519012])

        self.assertListEqual(
            list(agg_perf.drawdowns.index.strftime("%Y-%m-%d")),
            ['2018-12-03', '2018-12-04', '2018-12-05'])

        self.assertListEqual(
            agg_perf.drawdowns.tolist(),
            [0.0, -0.005406947999999967, -0.012608871878603933]
        )

        self.assertEqual(
            agg_perf.max_drawdown, -0.012608871878603933)

    def test_from_pnl_csv(self):

        self.maxDiff = None

        perf = DailyPerformance.from_pnl_csv("pnl.csv")
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
            perf.cum_pnl.to_dict(orient="list"),
           {'strategy-a': [0.0, 732.6318, 3195.9208000000003],
            'strategy-b': [0.0, 501.1911, 7035.3196]})

        self.assertListEqual(
            list(perf.commissions.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.commissions.to_dict(orient="list"),
            {'strategy-a': [0.0, 2.54e-05, 7.11e-05],
             'strategy-b': [0.0, 0.000179206, 5.55e-05]})

        self.assertDictEqual(
            perf.cagr.to_dict(),
            {'strategy-a': 1.5884135772875698, 'strategy-b': 7.013847123487736})

        self.assertDictEqual(
            perf.sharpe.to_dict(),
            {'strategy-a': 13.439089525191742, 'strategy-b': 10.255814111748151}
        )

        self.assertListEqual(
            list(perf.cum_returns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertDictEqual(
            perf.cum_returns.to_dict(orient="list"),
            {'strategy-a': [1.0, 1.00121226, 1.0052248083622792],
             'strategy-b': [1.0, 1.0008293, 1.011468946147319]}
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
            agg_perf.returns.tolist(),
            [0.0, 0.00204156, 0.014638520000000002])

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
            agg_perf.turnover.tolist(),
            [6.94e-18, 0.01, 0.02])

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
            agg_perf.pnl.tolist(),
            [0.0, 1233.8229000000001, 8997.4175])

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
            agg_perf.cum_pnl.tolist(),
           [0.0, 1233.8229000000001, 10231.240399999999])

        self.assertListEqual(
            list(agg_perf.commissions.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.commissions.tolist(),
            [0.0, 0.000204606, 0.0001266])

        self.assertEqual(
            agg_perf.cagr, 19.581032951701545)

        self.assertEqual(
            agg_perf.sharpe, 11.132759642908027)

        self.assertListEqual(
            list(agg_perf.cum_returns.index.strftime("%Y-%m-%d %H:%M:%S")),
            ['2019-01-21 00:00:00', '2019-01-22 00:00:00', '2019-01-23 00:00:00'])

        self.assertListEqual(
            agg_perf.cum_returns.tolist(),
            [1.0, 1.00204156, 1.0167099654168914]
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
            perf.sharpe.to_dict(),
            {'strategy-a': 13.439089525191742, 'strategy-b': 10.255814111748151}
        )

        perf = DailyPerformance.from_pnl_csv("pnl.csv", riskfree=0.02/252)
        self.assertDictEqual(
            perf.sharpe.to_dict(),
            {'strategy-a': 13.439089525191742, 'strategy-b': 10.255814111748151}
        )

    def test_compound(self):

        perf = DailyPerformance.from_moonshot_csv("backtest.csv")
        self.assertDictEqual(
            perf.cum_returns.to_dict(orient="list"),
            {'strategy-1': [0.997742875, 0.9973684510335559, 0.9949790474564671],
             'strategy-2': [1.00278717, 0.9977414688608159, 0.9929070442753247]}
        )

        perf = DailyPerformance.from_moonshot_csv("backtest.csv", compound=False)
        self.assertDictEqual(
            perf.cum_returns.to_dict(orient="list"),
            {'strategy-1': [0.997742875, 0.997367604, 0.994971896],
             'strategy-2': [1.00278717, 0.997755493, 0.992910125]}
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
            zscores.to_dict(orient="list"),
            {'strategy-1': [-0.5148664345304096, 1.152522272022025, -0.6376558374916148],
             'strategy-2': [1.1544487988426575, -0.5981044889340401, -0.5563443099086175]}
        )

        self.assertDictEqual(
            perf.returns.to_dict(orient="list"),
            {'strategy-1': [-0.002257125, -0.000375271, -0.002395708],
             'strategy-2': [0.00278717, -0.005031677, -0.004845368]}
        )

        perf = DailyPerformance.from_moonshot_csv("backtest.csv", trim_outliers=1.154)
        self.assertDictEqual(
            perf.returns.to_dict(orient="list"),
            {'strategy-1': [-0.002257125, -0.000375271, -0.002395708],
             'strategy-2': [0.0, -0.005031677, -0.004845368]}
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
            perf.benchmark_returns.tolist(),
            [0.0, 0.02237762237762242, -0.002540551104162625]
        )