# Copyright 2026 Chi-kwan Chan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Re-implement the L.A.Cosmic image cleaning algorithm with CuPy"""


from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cusmic")
except PackageNotFoundError:
    __version__ = "local-dev"


from .api import remove_cosmics
from .cleaner import Cleaner
from .image import Image

__all__ = ["Image", "Cleaner", "remove_cosmics"]
