"""
Categorical Predictors
======================
"""

###############################################################################
# The syntax for handling categorical predictors is **different** between standard regression models/two-stage-models (i.e. :code:`Lm` and :code:`Lm2`) and multi-level models (:code:`Lmer`) in :code:`pymer4`. This is because formula parsing is passed to R for :code:`Lmer` models, but handled by Python for other models. 

###############################################################################
# Lm and Lm2 Models
# -----------------
# :code:`Lm` and :code:`Lm2` models uses `patsy  <https://patsy.readthedocs.io/en/latest/>`_ to parse model formulae. Patsy is very powerful and has built-in support for handling categorical coding schemes (e.g. wrapping predictors in the :code:`C()` syntax). Patsy can also perform some pre-processing such as scaling and standardization. Here are some examples

# import basic libraries and sample data
import os
import pandas as pd
from pymer4.utils import get_resource_path
from pymer4.models import Lm
df = pd.read_csv(os.path.join(get_resource_path(), 'sample_data.csv'))

# IV3 is a categorical predictors with 3 levels 
# Estimate a model using Treatment contrasts (dummy-coding)
# with '1.0' as the reference level
model = Lm("DV ~ C(IV3, levels=[1.0, 0.5, 1.5])", data=df)
print(model.fit())

###############################################################################

# Now estimate using polynomial contrasts
model = Lm('DV ~ C(IV3, Poly)', data=df)
print(model.fit())


###############################################################################

# Sum-to-zero contrasts
model = Lm('DV ~ C(IV3, Sum)', data=df)
print(model.fit())


###############################################################################

# Moderation with IV2, but centering IV2 first
model = Lm('DV ~ center(IV2) * C(IV3, Sum)', data=df)
print(model.fit())

###############################################################################
# Please refer to the `patsy documentation <https://patsy.readthedocs.io/en/latest/categorical-coding.html>`_ for more details when working categorical predictors in :code:`Lm` or :code:`Lm2` models.

###############################################################################
# Lmer Models
# -----------
# :code:`Lmer()` models currently have support for handling categorical predictors in one of three ways based on how R's :code:`factor()` works:
# 
# - Dummy-coded factor levels (treatment contrasts) in which each model term is the difference between a factor level and a selected reference level
# - Orthogonal polynomial contrasts in which each model term is a polynomial contrast across factor levels (e.g. linear, quadratic, cubic, etc)
# - Custom contrasts for each level of a factor, which should be provide in the manner expected by R.
#
# To make re-parameterizing models easier, factor codings are passed as an argument to a model's :code:`.fit()` method unlike :code:`Lm` and :code:`Lm2` models. This obviates the need for adjusting data-frame properties as in R.

from pymer4.models import Lmer
# We're going to fit a multi-level logistic regression using the 
# dichotomous DV_l variable and the same categorical predictor (IV3)
# as before
model = Lmer('DV_l ~ IV3 + (IV3|Group)', data=df, family='binomial')

# Dummy-coding with '1.0' as the reference level
print(model.fit(factors={
    'IV3': ['1.0', '0.5', '1.5']
}))

