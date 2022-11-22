__version__ = "0.1.0a"

import logging
import os
import shutil
import sys
from pathlib import Path

PROJECT_PATH = Path(__file__).parent

SRC_PATH = PROJECT_PATH.parent
sys.path.append(str(SRC_PATH))

import numpy as np
from numpy.typing import NDArray

NDArray64 = NDArray[np.float64]
