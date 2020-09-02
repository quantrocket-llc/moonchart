# Copyright 2017 QuantRocket - All Rights Reserved
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

from setuptools import setup, find_packages
import versioneer

setup(name='quantrocket-moonchart',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Moonchart',
    long_description='Performance tear sheets for backtest analysis',
    url='https://www.quantrocket.com',
    author='QuantRocket LLC',
    author_email='support@quantrocket.com',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=[
        # pinning the matplotlib version is a temporary measure to avoid
        # the nuisance of small incremental matplotlib updates; it is not
        # a hard requirement and can be removed at an opportune time
        "matplotlib<=3.2.2",
        "pandas>=0.20",
        "seaborn",
        "quantrocket-client",
        "empyrical",
        "scipy",
    ]
)
