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

# To run: python -m unittest discover -s tests/ -p test*.py -t .

import unittest
import os
import pandas as pd
from quantrocket.moonshot import read_moonshot_csv
from moonchart.utils import intraday_to_daily

INTRADAY_AGGREGATE_RESULTS = {
    'fx-revert': {
        ('AbsExposure', '2018-12-18', '09:00:00'): 0.0,
        ('AbsExposure', '2018-12-18', '10:00:00'): 0.0,
        ('AbsExposure', '2018-12-18', '11:00:00'): 0.0,
        ('AbsExposure', '2018-12-19', '09:00:00'): 0.0,
        ('AbsExposure', '2018-12-19', '10:00:00'): 1.0,
        ('AbsExposure', '2018-12-19', '11:00:00'): 1.0,
        ('AbsWeight', '2018-12-18', '09:00:00'): 0.0,
        ('AbsWeight', '2018-12-18', '10:00:00'): 0.0,
        ('AbsWeight', '2018-12-18', '11:00:00'): 0.0,
        ('AbsWeight', '2018-12-19', '09:00:00'): 1.0,
        ('AbsWeight', '2018-12-19', '10:00:00'): 1.0,
        ('AbsWeight', '2018-12-19', '11:00:00'): 0.0,
        ('Benchmark', '2018-12-18', '09:00:00'): 1.136,
        ('Benchmark', '2018-12-18', '10:00:00'): 1.136465,
        ('Benchmark', '2018-12-18', '11:00:00'): 1.13606,
        ('Benchmark', '2018-12-19', '09:00:00'): 1.142945,
        ('Benchmark', '2018-12-19', '10:00:00'): 1.142125,
        ('Benchmark', '2018-12-19', '11:00:00'): 1.142185,
        ('Commission', '2018-12-18', '09:00:00'): 0.0,
        ('Commission', '2018-12-18', '10:00:00'): 0.0002,
        ('Commission', '2018-12-18', '11:00:00'): 0.00002,
        ('Commission', '2018-12-19', '09:00:00'): 0.0001,
        ('Commission', '2018-12-19', '10:00:00'): 0.0,
        ('Commission', '2018-12-19', '11:00:00'): 0.0,
        ('NetExposure', '2018-12-18', '09:00:00'): 0.0,
        ('NetExposure', '2018-12-18', '10:00:00'): 0.0,
        ('NetExposure', '2018-12-18', '11:00:00'): 0.0,
        ('NetExposure', '2018-12-19', '09:00:00'): 1.0,
        ('NetExposure', '2018-12-19', '10:00:00'): 1.5,
        ('NetExposure', '2018-12-19', '11:00:00'): 2.0,
        ('Return', '2018-12-18', '09:00:00'): 0.0,
        ('Return', '2018-12-18', '10:00:00'): 0.0,
        ('Return', '2018-12-18', '11:00:00'): 0.0,
        ('Return', '2018-12-19', '09:00:00'): 0.0010389911773893924,
        ('Return', '2018-12-19', '10:00:00'): -0.0004370168394709274,
        ('Return', '2018-12-19', '11:00:00'): -1.9934242843433506e-05,
        ('Slippage', '2018-12-18', '09:00:00'): 0.00001,
        ('Slippage', '2018-12-18', '10:00:00'): 0.00002,
        ('Slippage', '2018-12-18', '11:00:00'): 0.00001,
        ('Slippage', '2018-12-19', '09:00:00'): 0.00002,
        ('Slippage', '2018-12-19', '10:00:00'): 0.00001,
        ('Slippage', '2018-12-19', '11:00:00'): 0.00002,
        ('TotalHoldings', '2018-12-18', '09:00:00'): 0.0,
        ('TotalHoldings', '2018-12-18', '10:00:00'): 0.0,
        ('TotalHoldings', '2018-12-18', '11:00:00'): 0.0,
        ('TotalHoldings', '2018-12-19', '09:00:00'): 5.0,
        ('TotalHoldings', '2018-12-19', '10:00:00'): 5.0,
        ('TotalHoldings', '2018-12-19', '11:00:00'): 3.0,
        ('Turnover', '2018-12-18', '09:00:00'): 0.1,
        ('Turnover', '2018-12-18', '10:00:00'): 0.0,
        ('Turnover', '2018-12-18', '11:00:00'): 0.0,
        ('Turnover', '2018-12-19', '09:00:00'): 0.1,
        ('Turnover', '2018-12-19', '10:00:00'): 0.1,
        ('Turnover', '2018-12-19', '11:00:00'): 0.1}
}

INTRADAY_DETAILED_RESULTS = {
    'EUR.USD(12087792)': {
        ('AbsExposure', '2018-12-18', '09:00:00'): 0.0,
        ('AbsExposure', '2018-12-18', '10:00:00'): 0.0,
        ('AbsExposure', '2018-12-18', '11:00:00'): 0.0,
        ('AbsExposure', '2018-12-19', '09:00:00'): 0.2,
        ('AbsExposure', '2018-12-19', '10:00:00'): 0.2,
        ('AbsExposure', '2018-12-19', '11:00:00'): 0.2,
        ('AbsWeight', '2018-12-18', '09:00:00'): 0.0,
        ('AbsWeight', '2018-12-18', '10:00:00'): 0.0,
        ('AbsWeight', '2018-12-18', '11:00:00'): 0.0,
        ('AbsWeight', '2018-12-19', '09:00:00'): 0.2,
        ('AbsWeight', '2018-12-19', '10:00:00'): 0.2,
        ('AbsWeight', '2018-12-19', '11:00:00'): 0.2,
        ('Benchmark', '2018-12-18', '09:00:00'): 1.136,
        ('Benchmark', '2018-12-18', '10:00:00'): 1.136465,
        ('Benchmark', '2018-12-18', '11:00:00'): 1.13606,
        ('Benchmark', '2018-12-19', '09:00:00'): 1.142945,
        ('Benchmark', '2018-12-19', '10:00:00'): 1.142125,
        ('Benchmark', '2018-12-19', '11:00:00'): 1.142185,
        ('Commission', '2018-12-18', '09:00:00'): 0.0,
        ('Commission', '2018-12-18', '10:00:00'): 0.0,
        ('Commission', '2018-12-18', '11:00:00'): 0.00001,
        ('Commission', '2018-12-19', '09:00:00'): 0.0,
        ('Commission', '2018-12-19', '10:00:00'): 0.00002,
        ('Commission', '2018-12-19', '11:00:00'): 0.0,
        ('Mavg', '2018-12-18', '09:00:00'): None,
        ('Mavg', '2018-12-18', '10:00:00'): None,
        ('Mavg', '2018-12-18', '11:00:00'): None,
        ('Mavg', '2018-12-19', '09:00:00'): 1.1369470000000004,
        ('Mavg', '2018-12-19', '10:00:00'): 1.1370885000000004,
        ('Mavg', '2018-12-19', '11:00:00'): 1.1372462000000003,
        ('NetExposure', '2018-12-18', '09:00:00'): 0.0,
        ('NetExposure', '2018-12-18', '10:00:00'): 0.0,
        ('NetExposure', '2018-12-18', '11:00:00'): 0.0,
        ('NetExposure', '2018-12-19', '09:00:00'): 0.2,
        ('NetExposure', '2018-12-19', '10:00:00'): 0.2,
        ('NetExposure', '2018-12-19', '11:00:00'): 0.2,
        ('Return', '2018-12-18', '09:00:00'): -0.0,
        ('Return', '2018-12-18', '10:00:00'): -0.0,
        ('Return', '2018-12-18', '11:00:00'): -0.0,
        ('Return', '2018-12-19', '09:00:00'): 0.0004902863658290624,
        ('Return', '2018-12-19', '10:00:00'): -0.00014348896928548794,
        ('Return', '2018-12-19', '11:00:00'): 1.0506730874437764e-05,
        ('Signal', '2018-12-18', '09:00:00'): 0.0,
        ('Signal', '2018-12-18', '10:00:00'): 0.0,
        ('Signal', '2018-12-18', '11:00:00'): 0.0,
        ('Signal', '2018-12-19', '09:00:00'): 1.0,
        ('Signal', '2018-12-19', '10:00:00'): 1.0,
        ('Signal', '2018-12-19', '11:00:00'): 1.0,
        ('Slippage', '2018-12-18', '09:00:00'): 0.0,
        ('Slippage', '2018-12-18', '10:00:00'): 0.0,
        ('Slippage', '2018-12-18', '11:00:00'): 0.0,
        ('Slippage', '2018-12-19', '09:00:00'): 0.0,
        ('Slippage', '2018-12-19', '10:00:00'): 0.00002,
        ('Slippage', '2018-12-19', '11:00:00'): 0.00002,
        ('TotalHoldings', '2018-12-18', '09:00:00'): 0.0,
        ('TotalHoldings', '2018-12-18', '10:00:00'): 0.0,
        ('TotalHoldings', '2018-12-18', '11:00:00'): 0.0,
        ('TotalHoldings', '2018-12-19', '09:00:00'): 1.0,
        ('TotalHoldings', '2018-12-19', '10:00:00'): 2.0,
        ('TotalHoldings', '2018-12-19', '11:00:00'): 1.0,
        ('Turnover', '2018-12-18', '09:00:00'): 0.0,
        ('Turnover', '2018-12-18', '10:00:00'): 0.0,
        ('Turnover', '2018-12-18', '11:00:00'): 0.0,
        ('Turnover', '2018-12-19', '09:00:00'): 0.1,
        ('Turnover', '2018-12-19', '10:00:00'): 0.1,
        ('Turnover', '2018-12-19', '11:00:00'): 0.1,
        ('Weight', '2018-12-18', '09:00:00'): 0.0,
        ('Weight', '2018-12-18', '10:00:00'): 0.0,
        ('Weight', '2018-12-18', '11:00:00'): 0.0,
        ('Weight', '2018-12-19', '09:00:00'): 0.1,
        ('Weight', '2018-12-19', '10:00:00'): 0.15,
        ('Weight', '2018-12-19', '11:00:00'): 0.2},
    'GBP.USD(12087797)': {
        ('AbsExposure', '2018-12-18', '09:00:00'): 0.0,
        ('AbsExposure', '2018-12-18', '10:00:00'): 0.0,
        ('AbsExposure', '2018-12-18', '11:00:00'): 0.0,
        ('AbsExposure', '2018-12-19', '09:00:00'): 0.2,
        ('AbsExposure', '2018-12-19', '10:00:00'): 0.2,
        ('AbsExposure', '2018-12-19', '11:00:00'): 0.2,
        ('AbsWeight', '2018-12-18', '09:00:00'): 0.0,
        ('AbsWeight', '2018-12-18', '10:00:00'): 0.0,
        ('AbsWeight', '2018-12-18', '11:00:00'): 0.0,
        ('AbsWeight', '2018-12-19', '09:00:00'): 0.2,
        ('AbsWeight', '2018-12-19', '10:00:00'): 0.2,
        ('AbsWeight', '2018-12-19', '11:00:00'): 0.2,
        ('Benchmark', '2018-12-18', '09:00:00'): None,
        ('Benchmark', '2018-12-18', '10:00:00'): None,
        ('Benchmark', '2018-12-18', '11:00:00'): None,
        ('Benchmark', '2018-12-19', '09:00:00'): None,
        ('Benchmark', '2018-12-19', '10:00:00'): None,
        ('Benchmark', '2018-12-19', '11:00:00'): None,
        ('Commission', '2018-12-18', '09:00:00'): 0.0,
        ('Commission', '2018-12-18', '10:00:00'): 0.0,
        ('Commission', '2018-12-18', '11:00:00'): 0.0,
        ('Commission', '2018-12-19', '09:00:00'): 0.0,
        ('Commission', '2018-12-19', '10:00:00'): 0.00001,
        ('Commission', '2018-12-19', '11:00:00'): 0.00002,
        ('Mavg', '2018-12-18', '09:00:00'): None,
        ('Mavg', '2018-12-18', '10:00:00'): None,
        ('Mavg', '2018-12-18', '11:00:00'): None,
        ('Mavg', '2018-12-19', '09:00:00'): 1.2640600000000104,
        ('Mavg', '2018-12-19', '10:00:00'): 1.2641065000000105,
        ('Mavg', '2018-12-19', '11:00:00'): 1.2642211000000103,
        ('NetExposure', '2018-12-18', '09:00:00'): 0.0,
        ('NetExposure', '2018-12-18', '10:00:00'): 0.0,
        ('NetExposure', '2018-12-18', '11:00:00'): 0.0,
        ('NetExposure', '2018-12-19', '09:00:00'): 0.1,
        ('NetExposure', '2018-12-19', '10:00:00'): 0.2,
        ('NetExposure', '2018-12-19', '11:00:00'): 0.3,
        ('Return', '2018-12-18', '09:00:00'): -0.0,
        ('Return', '2018-12-18', '10:00:00'): -0.0,
        ('Return', '2018-12-18', '11:00:00'): 0.0,
        ('Return', '2018-12-19', '09:00:00'): 0.0007272756081426303,
        ('Return', '2018-12-19', '10:00:00'): -0.0004057355535645791,
        ('Return', '2018-12-19', '11:00:00'): 0.0003899498918343625,
        ('Signal', '2018-12-18', '09:00:00'): 0.0,
        ('Signal', '2018-12-18', '10:00:00'): 0.0,
        ('Signal', '2018-12-18', '11:00:00'): 0.0,
        ('Signal', '2018-12-19', '09:00:00'): 1.0,
        ('Signal', '2018-12-19', '10:00:00'): 1.0,
        ('Signal', '2018-12-19', '11:00:00'): 1.0,
        ('Slippage', '2018-12-18', '09:00:00'): 0.0,
        ('Slippage', '2018-12-18', '10:00:00'): 0.0,
        ('Slippage', '2018-12-18', '11:00:00'): 0.0,
        ('Slippage', '2018-12-19', '09:00:00'): 0.00001,
        ('Slippage', '2018-12-19', '10:00:00'): 0.0,
        ('Slippage', '2018-12-19', '11:00:00'): 0.00002,
        ('TotalHoldings', '2018-12-18', '09:00:00'): 0.0,
        ('TotalHoldings', '2018-12-18', '10:00:00'): 0.0,
        ('TotalHoldings', '2018-12-18', '11:00:00'): 0.0,
        ('TotalHoldings', '2018-12-19', '09:00:00'): 1.0,
        ('TotalHoldings', '2018-12-19', '10:00:00'): 1.0,
        ('TotalHoldings', '2018-12-19', '11:00:00'): 1.0,
        ('Turnover', '2018-12-18', '09:00:00'): 0.0,
        ('Turnover', '2018-12-18', '10:00:00'): 0.0,
        ('Turnover', '2018-12-18', '11:00:00'): 0.0,
        ('Turnover', '2018-12-19', '09:00:00'): 0.0,
        ('Turnover', '2018-12-19', '10:00:00'): 0.0,
        ('Turnover', '2018-12-19', '11:00:00'): 0.0,
        ('Weight', '2018-12-18', '09:00:00'): 0.0,
        ('Weight', '2018-12-18', '10:00:00'): 0.0,
        ('Weight', '2018-12-18', '11:00:00'): 0.0,
        ('Weight', '2018-12-19', '09:00:00'): 0.2,
        ('Weight', '2018-12-19', '10:00:00'): 0.2,
        ('Weight', '2018-12-19', '11:00:00'): 0.2}
 }

class IntradayToDailyTestCase(unittest.TestCase):
    """
    Test cases for `moonchart.utils.intraday_to_daily`.
    """

    def tearDown(self):
        if os.path.exists("results.csv"):
            os.remove("results.csv")


    def test_aggregate_intraday_to_daily(self):

        results = pd.DataFrame.from_dict(INTRADAY_AGGREGATE_RESULTS)
        results.index.set_names(["Field","Date", "Time"], inplace=True)
        results.to_csv("results.csv")

        intraday_results = read_moonshot_csv("results.csv")

        daily_results = intraday_to_daily(intraday_results)

        daily_results = daily_results.reset_index()
        daily_results.loc[:, "Date"] = daily_results.Date.dt.strftime("%Y-%m-%d")
        daily_results = daily_results.set_index(["Field", "Date"])

        self.assertDictEqual(
            daily_results.to_dict(),
            {'fx-revert': {
                # max
                ('AbsExposure', '2018-12-18'): 0.0,
                ('AbsExposure', '2018-12-19'): 1.0,
                # max
                ('AbsWeight', '2018-12-18'): 0.0,
                ('AbsWeight', '2018-12-19'): 1.0,
                # last
                ('Benchmark', '2018-12-18'): 1.13606,
                ('Benchmark', '2018-12-19'): 1.142185,
                # sum
                ('Commission', '2018-12-18'): 0.00022,
                ('Commission', '2018-12-19'): 0.0001,
                # mean
                ('NetExposure', '2018-12-18'): 0.0,
                ('NetExposure', '2018-12-19'): 1.5,
                # sum
                ('Return', '2018-12-18'): 0.0,
                ('Return', '2018-12-19'): 0.0005820400950750315,
                # sum
                ('Slippage', '2018-12-18'): 4e-05,
                ('Slippage', '2018-12-19'): 5.000000000000001e-05,
                # max
                ('TotalHoldings', '2018-12-18'): 0.0,
                ('TotalHoldings', '2018-12-19'): 5.0,
                # sum
                ('Turnover', '2018-12-18'): 0.1,
                ('Turnover', '2018-12-19'): 0.30000000000000004}}
        )

    def test_detailed_intraday_to_daily(self):

        results = pd.DataFrame.from_dict(INTRADAY_DETAILED_RESULTS)
        results.index.set_names(["Field","Date", "Time"], inplace=True)
        results.to_csv("results.csv")

        intraday_results = read_moonshot_csv("results.csv")

        daily_results = intraday_to_daily(intraday_results)

        daily_results = daily_results.reset_index()
        daily_results.loc[:, "Date"] = daily_results.Date.dt.strftime("%Y-%m-%d")
        daily_results = daily_results.set_index(["Field", "Date"])

        daily_results = daily_results.where(daily_results.notnull(), None)

        self.assertDictEqual(
            daily_results.to_dict(),
            {'EUR.USD(12087792)': { # max
                                   ('AbsExposure', '2018-12-18'): 0.0,
                                   ('AbsExposure', '2018-12-19'): 0.2,
                                   # max
                                   ('AbsWeight', '2018-12-18'): 0.0,
                                   ('AbsWeight', '2018-12-19'): 0.2,
                                   # last
                                   ('Benchmark', '2018-12-18'): 1.13606,
                                   ('Benchmark', '2018-12-19'): 1.142185,
                                   # sum
                                   ('Commission', '2018-12-18'): 1e-05,
                                   ('Commission', '2018-12-19'): 2e-05,
                                   # mean
                                   ('NetExposure', '2018-12-18'): 0.0,
                                   ('NetExposure', '2018-12-19'): 0.20000000000000004,
                                   # sum
                                   ('Return', '2018-12-18'): 0.0,
                                   ('Return', '2018-12-19'): 0.00035730412741801225,
                                   # sum
                                   ('Slippage', '2018-12-18'): 0.0,
                                   ('Slippage', '2018-12-19'): 4e-05,
                                   # max
                                   ('TotalHoldings', '2018-12-18'): 0.0,
                                   ('TotalHoldings', '2018-12-19'): 2.0,
                                   # sum
                                   ('Turnover', '2018-12-18'): 0.0,
                                   ('Turnover', '2018-12-19'): 0.30000000000000004,
                                   # mean
                                   ('Weight', '2018-12-18'): 0.0,
                                   ('Weight', '2018-12-19'): 0.15},
             'GBP.USD(12087797)': {# max
                                   ('AbsExposure', '2018-12-18'): 0.0,
                                   ('AbsExposure', '2018-12-19'): 0.2,
                                   # max
                                   ('AbsWeight', '2018-12-18'): 0.0,
                                   ('AbsWeight', '2018-12-19'): 0.2,
                                   # last
                                   ('Benchmark', '2018-12-18'): None,
                                   ('Benchmark', '2018-12-19'): None,
                                   # sum
                                   ('Commission', '2018-12-18'): 0.0,
                                   ('Commission', '2018-12-19'): 3.0000000000000004e-05,
                                   # mean
                                   ('NetExposure', '2018-12-18'): 0.0,
                                   ('NetExposure', '2018-12-19'): 0.20000000000000004,
                                   # sum
                                   ('Return', '2018-12-18'): 0.0,
                                   ('Return', '2018-12-19'): 0.0007114899464124138,
                                   # sum
                                   ('Slippage', '2018-12-18'): 0.0,
                                   ('Slippage', '2018-12-19'): 3.0000000000000004e-05,
                                   # max
                                   ('TotalHoldings', '2018-12-18'): 0.0,
                                   ('TotalHoldings', '2018-12-19'): 1.0,
                                   # sum
                                   ('Turnover', '2018-12-18'): 0.0,
                                   ('Turnover', '2018-12-19'): 0.0,
                                   # mean
                                   ('Weight', '2018-12-18'): 0.0,
                                   ('Weight', '2018-12-19'): 0.20000000000000004}}
        )
