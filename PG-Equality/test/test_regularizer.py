'''
# File: test_regularizer.py
# Project: First order method for CCO
# Description: The code is to test regularizer.py.
# ----------	---	----------------------------------------------------------
'''

import numpy as np
import sys
sys.path.append("../")
from src.funcs.regularizer import L1

if __name__ == "__main__":
    r = L1(penalty=1)
    x = [1,1]
    assert r.obj(x) == 2
