"""
5. Additional Statistics Functions
==================================
:code:`pymer4` also comes with some flexible routines for various statistical operations such as permutation testing, bootstrapping of arbitrary functions and equivalence testing. Here are a few examples
"""

###############################################################################
#  Permutation Tests
# -----------------
# :code:`pymer4` can compute a wide variety of one and two-sample permutation tests including mean differences, t-statistics, effect size comparisons, and correlations

# Import numpy and set random number generator
import numpy as np
np.random.seed(10)
# Import stats functions
from pymer4.stats import perm_test

# Generate two samples of data: X (M~2, SD~10, N=100) and Y (M~2.5, SD~1, N=100)
x = np.random.normal(loc=2, size=100)
y = np.random.normal(loc=2.5, size=100)

# Between groups t-test
result = perm_test(x, y, stat="tstat", n_perm=500, n_jobs=1)
print(result)

###############################################################################

# Spearman rank correlation
result = perm_test(x, y, stat="spearmanr", n_perm=500, n_jobs=1)
print(result)

###############################################################################
#  Bootstrap Comparisons
# ----------------------
# :code:`pymer4` can compute a bootstrap comparison using any arbitrary function that takes as input either 1 or 2, 1d numpy arrays and returns a single value.

# Import stats function
from pymer4.stats import boot_func


# Define a simple function for a median difference test
def med_diff(x, y):
    return np.median(x) - np.median(y)


# Between groups median test with resampling
result = boot_func(x, y, func=med_diff)
print(result)

###############################################################################
# TOST Equivalence Tests
# ----------------------
# :code:`pymer4` also has experimental support for two-one-sided equivalence tests

# Import stats function
from pymer4.stats import tost_equivalence

# Generate some data
lower, upper = -0.1, 0.1
x, y = np.random.normal(0.145, 0.025, 35), np.random.normal(0.16, 0.05, 17)
result = tost_equivalence(x, y, lower, upper, plot=True)
print(result)

