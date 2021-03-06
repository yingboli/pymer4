.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_example_04_simulating_data.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_example_04_simulating_data.py:


4. Simulating Data
==================
:code:`pymer4` comes with some easy-to-use functions for simulating data that can be modeled with :code:`Lm` and multi-level data that can be modeled with :code:`Lmer` or :code:`Lm2`. These functions can be found in the :code:`pymer4.simulate` module and are aptly named: :code:`simulate_lm()` and :code:`simulate_lmm()` respectively.

:code:`pymer4` gives you a lot of control over what you want your data to look like by setting properties such as:

- Number of data points and number of coefficients
- Specific coefficient values
- Means and standard deviations of predictors
- Correlations between predictors
- Amount of error (noise) in the data
- Number of groups/clusters (multi-level data only)
- Variance of random effects (multi-level data only)

Generating standard regression data
-----------------------------------
Generating data for a standard regression returns a pandas dataframe with outcome and predictor variables ready for use with :code:`Lm()`, along with an array of coefficients used to produce the data.

Let's generate 500 observations, with coefficient values of: 1.2, -40.1, and 3. We also have an intercept with a value of 100. The means of the columns of our design matrix (i.e. means of the predictors) will be: 10, 30, and 1. We'll also add noise from a normal distribution with mean = 0, and sd = 5. Any correlations between predictors are purely random.


.. code-block:: default


    # Import the simulation function
    from pymer4.simulate import simulate_lm
    # Also fix the random number generator for reproducibility
    import numpy as np
    np.random.seed(10)

    data, b = simulate_lm(
        500, 3, coef_vals=[100, 1.2, -40.1, 3], mus=[10, 30, 1], noise_params=(0, 5)
    )
    print(f"True coefficients:\n{b}\n")
    print(f"Data:\n{data.head()}")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True coefficients:
    [100, 1.2, -40.1, 3]

    Data:
                DV        IV1        IV2       IV3
    0 -1114.325941  11.331587  30.715279 -0.545400
    1 -1116.406768   9.991616  30.621336  0.279914
    2 -1088.455691  10.265512  30.108549  1.004291
    3 -1092.013632   9.825400  30.433026  2.203037
    4 -1124.899182   9.034934  31.028274  1.228630



Here are some checks you might do to make sure the data were correctly generated:

Check the means of predictors


.. code-block:: default

    print(data.iloc[:, 1:].mean(axis=0))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    IV1    10.002923
    IV2    30.039709
    IV3     0.962177
    dtype: float64



Check correlations between predictors


.. code-block:: default

    print(data.iloc[:, 1:].corr())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

              IV1       IV2       IV3
    IV1  1.000000 -0.013148 -0.010051
    IV2 -0.013148  1.000000 -0.051630
    IV3 -0.010051 -0.051630  1.000000



Check coefficient recovery


.. code-block:: default

    from pymer4.models import Lm

    model = Lm("DV ~ IV1+IV2+IV3", data=data)
    model.fit(summarize=False)
    print(model.coefs.loc[:, "Estimate"])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Intercept    95.474548
    IV1           1.342881
    IV2         -40.001760
    IV3           2.859270
    Name: Estimate, dtype: float64



You have the option of being as general or specific as you like when generating data. Here's a simpler example that generates 100 observations with 5 predictors from a standard normal distribution, i.e. mean = 0, sd = 1 with random correlations between predictors. :code:`pymer4` will randomly decide what to set the coefficient values to.


.. code-block:: default


    data, b = simulate_lm(100, 5)
    print(f"True coefficients:\n{b}\n")
    print(f"Data:\n{data.head()}")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True coefficients:
    [0.05682538 0.04259271 0.63572183 0.2399937  0.08991266 0.17923857]

    Data:
             DV       IV1       IV2       IV3       IV4       IV5
    0 -1.619562 -0.063833 -0.471785 -0.419493  1.270657 -1.576390
    1  1.493992  0.670564  1.008049  1.803014 -0.040395 -0.621471
    2 -1.630406 -1.527920  0.199663 -1.006917  0.062326 -0.190250
    3 -0.315245  0.424936 -0.171909 -0.144126  1.227489  0.078798
    4  1.911261  1.242033 -0.811868  0.446330  0.356810 -0.437578



Generating multi-level regression data
--------------------------------------
Generating data for a multi-level regression is just as simple and returns a pandas dataframe with outcome and predictor variables ready for use with :code:`Lmer()`, another dataframe with group/cluster level coefficients (i.e. BLUPs), and a vector of population-level coefficients.

Here's an example generating 5000 observations, organized as 100 groups with 50 observations each. We'll have three predictors with the coefficients: 1.8, -2, and 10. We also have an intercept with a coefficient of 4. The means of the columns of our design matrix (i.e. means of the predictors) will be: 10, 30, and 2. We'll also introduce correlations between our predictors of with a mean r of .15. We'll leave the default of standard normal noise i.e., mean = 0, and sd = 1.


.. code-block:: default


    from pymer4.simulate import simulate_lmm

    num_obs = 50
    num_coef = 3
    num_grps = 100
    mus = [10.0, 30.0, 2.0]
    coef_vals = [4.0, 1.8, -2, 10]
    corrs = 0.15

    data, blups, b = simulate_lmm(
        num_obs, num_coef, num_grps, coef_vals=coef_vals, mus=mus, corrs=corrs
    )

    print(f"True coefficients:\n{b}\n")
    print(f"BLUPs:\n{blups.head()}\n")
    print(f"Data:\n{data.head()}\n")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    True coefficients:
    [4.0, 1.8, -2, 10]

    BLUPs:
          Intercept       IV1       IV2        IV3
    Grp1   4.118082  1.908896 -1.769091   9.887560
    Grp2   4.250422  1.898551 -1.513031  10.359999
    Grp3   4.076250  1.858520 -2.267093  10.168399
    Grp4   3.830477  1.776946 -1.921247   9.583227
    Grp5   4.141466  2.170102 -1.892564  10.349354

    Data:
              DV        IV1        IV2       IV3  Group
    0  -4.179066   9.383356  29.476310  2.438898    1.0
    1   8.983399  12.129908  31.362946  3.859619    1.0
    2 -13.442347  10.061723  29.302197  1.580586    1.0
    3 -10.241627  10.758237  29.259286  1.631702    1.0
    4 -15.502489  11.585787  30.199303  1.076930    1.0




Again here are some checks you might do to make sure the data were correctly generated (by default lmm data will generally be a bit noisier due to within and across group/cluster variance; see the API for how to customize this):


.. code-block:: default


    # Group the data before running checks
    group_data = data.groupby("Group")







Check mean of predictors within each group


.. code-block:: default

    print(group_data.apply(lambda grp: grp.iloc[:, 1:-1].mean(axis=0)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

                 IV1        IV2       IV3
    Group                                
    1.0     9.901321  30.039194  1.758267
    2.0     9.976000  30.104749  1.984167
    3.0    10.222086  30.194326  1.905938
    4.0     9.879292  30.215769  2.130761
    5.0     9.903163  30.274854  1.941497
    ...          ...        ...       ...
    96.0    9.943912  29.950404  1.952312
    97.0   10.047164  29.978932  2.231869
    98.0    9.997547  30.018299  2.205165
    99.0   10.213984  30.044085  1.965605
    100.0   9.965338  30.120661  1.870400

    [100 rows x 3 columns]



Check correlations between predictors within each group


.. code-block:: default

    print(group_data.apply(lambda grp: grp.iloc[:, 1:-1].corr()))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

                    IV1       IV2       IV3
    Group                                  
    1.0   IV1  1.000000  0.272855  0.303139
          IV2  0.272855  1.000000  0.134635
          IV3  0.303139  0.134635  1.000000
    2.0   IV1  1.000000  0.079445  0.373448
          IV2  0.079445  1.000000  0.002340
    ...             ...       ...       ...
    99.0  IV2  0.113312  1.000000  0.235816
          IV3  0.055161  0.235816  1.000000
    100.0 IV1  1.000000  0.317120  0.261968
          IV2  0.317120  1.000000  0.139132
          IV3  0.261968  0.139132  1.000000

    [300 rows x 3 columns]



Check coefficient recovery


.. code-block:: default

    from pymer4.models import Lmer

    model = Lmer('DV ~ IV1+IV2+IV3 + (1|Group)', data=data)
    model.fit(summarize=False)
    print(model.coefs.loc[:, "Estimate"])







.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (Intercept)     4.082829
    IV1             1.845101
    IV2            -2.007044
    IV3            10.023242
    Name: Estimate, dtype: float64




.. _sphx_glr_download_auto_examples_example_04_simulating_data.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: example_04_simulating_data.py <example_04_simulating_data.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: example_04_simulating_data.ipynb <example_04_simulating_data.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
