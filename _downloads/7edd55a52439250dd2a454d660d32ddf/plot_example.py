"""
Basic Usage Guide
=================
"""

###############################################################################
#.. note::
#   :code:`print()` statements in the examples below are only so these docs render at this site. They are not required for actual usage
#

###############################################################################
# All :code:`pymer4` models operate on long-format `pandas dataframes <https://pandas.pydata.org/pandas-docs/version/0.25/index.html/>`_. dataframes. These dataframes should contain columns for a dependent variable (DV), independent variable(s) (IV), and optionally a column for a group/cluster identifiers (Group). :code:`pymer4` comes with sample data for testing purposes which we'll utilize here.
# For testing purposes this sample data has: 
# 
# - Two kinds of dependent variables: *DV* (continuous), *DV_l* (dichotomous)
# - Two kinds of independent variables: *IV1* (continuous), *IV2* (categorical).
# - One grouping variable for multi-level modeling: *Group*.
# 
# Let's check it out below:

# import some basic libraries
import os
import pandas as pd

# get utility function for sample data path
from pymer4.utils import get_resource_path

# Load and checkout sample data
df = pd.read_csv(os.path.join(get_resource_path(), 'sample_data.csv'))
print(df.head())

###############################################################################
# Standard regression models
# ------------------------------------
# Fitting a standard regression model is similar to using :code:`lm()` in R. The difference is that :code:`pymer4` uses the object-oriented design-style of other scientific python libraries like `scikit-learn <https://scikit-learn.org/stable/index.html/>`_. 
# This simply means initializing a model object with a formula and some data and then calling its :code:`.fit()` method.
#
# By default the output of :code:`.fit()` has been formated to be a blend of :code:`summary()` in R and :code:`.summary()` from `statsmodels <http://www.statsmodels.org/dev/index.html/>`_. This includes metadata about the model, data, and overall fit as well as estimates and inference results of model terms.  

# Import the linear regression model class
from pymer4.models import Lm

# Initialize model using 2 predictors
model = Lm('DV ~ IV1 + IV3', data=df)

# Fit it
print(model.fit())

###############################################################################
# Multi-level models
# ----------------------------
# Fitting a multi-level model works similarly, and the corresponding output is formatted to be very similar to output of :code:`summary()` in R.

# Import the lmm model class
from pymer4.models import Lmer

# Initialize model instance using 1 predictor with random intercepts and slopes
model = Lmer('DV ~ IV2 + (IV2|Group)', data=df)

# Fit it
print(model.fit())




