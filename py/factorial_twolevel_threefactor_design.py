#!/usr/bin/env python
# coding: utf-8

# # A Two-Level, Three-Factor Full Factorial Design
# 
# <br />
# <br />
# <br />
# 
# ### Table of Contents
# 
# * [Introduction](#intro)
# * Factorial Experimental Design:
#  * [Two-Level Three-Factor Full Factorial Design](#twolevelfactorial)
#  * [Design of the Experiment](#designexperiment)
#  * [Inputs and Responses](#inputs_responses)
# * Effects and Interactions:
#  * [Computing Main Effects](#computing_main_effects)
#  * [Analyzing Main Effects](#analyzing_main_effects)
#  * [Two Way Interactions](#twowayinteractions)
#  * [Analyzing Two Way Interactions](#analyzing_twowayinteractions)
#  * [Three Way Interactions](#threewayinteractions)
#  * [Analyzing Three Way Interactions](#analyzing_threewayinteractions)
#  * [Fitting a Polynomial Response Surface](#fitting_responsesurface)
# * Uncertainty:
#  * [The Impact of Uncertainty](#uncertainty)
#  * [Uncertainty Quantification: A Factory Example](#uncertainty_example)
#  * [Uncertainty Numbers](#uncertainty_numbers)
#  * [Uncertainty Measurements](#uncertainty_measurements)
#  * [Accounting for Uncertainty in the Model](#uncertainty_accounting)
# * [Discussion](#discussion)
# * [Conclusion](#conclusion)
# 
# <br />
# <br />
# <br />
# 

# <a name="intro"></a>
# ## Introduction
# 
# As with other notebooks in this repository, this notebook follows, more or less closely, content from Box and Draper's _Empirical Model-Building and Response Surfaces_ (Wiley, 1984). This content is covered by Chapter 4 of Box and Draper.
# 
# In this notebook, we'll carry out an anaylsis of a full factorial design, and show how we can obtain inforomation about a system and its responses, and a quantifiable range of certainty about those values. This is the fundamental idea behind empirical model-building and allows us to construct cheap and simple models to represent complex, nonlinear systems.
# 
# Once we've nailed this down for simple models and small numbers of inputs and responses, we can expand on it, use more complex models, and link this material with machine learning algorithms.

# We'll start by importing numpy for numerical analysis, and pandas for convenient data containers.

# In[1]:


import pandas as pd
import numpy as np
from numpy.random import rand


# Box and Draper cover different experimental design methods in the book, but begin with the simplest type of factorial design in Chapter 4: a full factorial design with two levels. A factorial experimental design is appropriate for exploratory stages, when the effects of variables or their interactions on a system response are poorly understood or not quantifiable.

# <a name="twolevelfactorial"></a>
# ## Two-Level Full Factorial Design
# 
# The analysis begins with a two-level, three-variable experimental design - also written $2^3$, with $n=2$ levels for each factor, $k=3$ different factors. We start by encoding each fo the three variables to something generic: $(x_1,x_2,x_3)$. A dataframe with input variable values is then populated.

# In[2]:


inputs_labels = {'x1' : 'Length of specimen (mm)',
                 'x2' : 'Amplitude of load cycle (mm)',
                 'x3' : 'Load (g)'}

dat = [('x1',250,350),
       ('x2',8,10),
       ('x3',40,50)]

inputs_df = pd.DataFrame(dat,columns=['index','low','high'])
inputs_df = inputs_df.set_index(['index'])
inputs_df['label'] = inputs_df.index.map( lambda z : inputs_labels[z] )

inputs_df


# Next, we encode the variable values. For an arbitrary variable value $\phi_1$, the value of the variable can be coded to be between -1 and 1 according to the formula:
# 
# $$
# x_i = \dfrac{ \phi_i - \mbox{avg }(\phi) }{ \mbox{span }(\phi) }
# $$
# 
# where the average and the span of the variable $\phi_i$ are defined as:
# 
# $$
# \mbox{avg }(\phi) = \left( \dfrac{ \phi_{\text{high}} + \phi_{\text{low}} }{2} \right)
# $$
# 
# $$
# \mbox{span }(\phi) = \left( \dfrac{ \phi_{\text{high}} - \phi_{\text{low}} }{2} \right)
# $$
# 

# In[3]:


inputs_df['average']      = inputs_df.apply( lambda z : ( z['high'] + z['low'])/2 , axis=1)
inputs_df['span']         = inputs_df.apply( lambda z : ( z['high'] - z['low'])/2 , axis=1)

inputs_df['encoded_low']  = inputs_df.apply( lambda z : ( z['low']  - z['average'] )/( z['span'] ), axis=1)
inputs_df['encoded_high'] = inputs_df.apply( lambda z : ( z['high'] - z['average'] )/( z['span'] ), axis=1)

inputs_df = inputs_df.drop(['average','span'],axis=1)

inputs_df


# <a name="designexperiment"></a>
# ## Design of the Experiment
# 
# While everything preceding this point is important to state, to make sure we're being consistent and clear about our problem statement and assumptions, nothing preceding this point is particularly important to understanding how experimental design works. This is simply illustrating the process of transforming one's problem from a problem-specific problem space to a more general problem space.

# <a name="inputs_responses"></a>
# ## Inputs and Responses
# 
# Box and Draper present the results (observed outcomes) of a $2^3$ factorial experiment. The $2^3$ comes from the fact that there are 2 levels for each variable (-1 and 1) and three variables (x1, x2, and x3). The observed, or output, variable is the number of cycles to failure for a particular piece of machinery; this variable is more conveniently cast as a logarithm, as it can be a very large number.
# 
# Each observation data point consists of three input variable values and an output variable value, $(x_1, x_2, x_3, y)$, and can be thought of as a point in 3D space $(x_1,x_2,x_3)$ with an associated point value of $y$. Alternatively, this might be thought of as a point in 4D space (the first three dimensions are the location in 3D space where the point will appear, and the $y$ value is when it will actually appear). 
# 
# The input variable values consist of all possible input value combinations, which we can produce using the itertools module:

# In[4]:


import itertools
encoded_inputs = list( itertools.product([-1,1],[-1,1],[-1,1]) )
encoded_inputs


# Now we implement the observed outcomes; as we mentioned, these numbers are large (hundreds or thousands of cycles), and are more conveniently scaled by taking $\log_{10}()$ (which will rescale them to be integers between 1 and 4).

# In[5]:


results = [(-1, -1, -1, 674),
           ( 1, -1, -1, 3636),
           (-1,  1, -1, 170),
           ( 1,  1, -1, 1140),
           (-1, -1,  1, 292),
           ( 1, -1,  1, 2000),
           (-1,  1,  1, 90),
           (  1, 1,  1, 360)]

results_df = pd.DataFrame(results,columns=['x1','x2','x3','y'])
results_df['logy'] = results_df['y'].map( lambda z : np.log10(z) )
results_df


# The variable `inputs_df` contains all input variables for the expeirment design, and `results_df` contains the inputs and responses for the experiment design; these variables are the encoded levels. To obtain the original, unscaled values, which allows us to check what experiments must be run, we can always convert the dataframe back to its originals by defining a function to un-apply the scaling equation. This is as simple as finding 

# In[7]:


real_experiment = results_df

var_labels = []
for var in ['x1','x2','x3']:
    var_label = inputs_df.ix[var]['label']
    var_labels.append(var_label)
    real_experiment[var_label] = results_df.apply(
        lambda z : inputs_df.ix[var]['low'] if z[var]<0 else inputs_df.ix[var]['high'] , 
        axis=1)

print("The values of each real variable in the experiment:")
real_experiment[var_labels]


# <a name="computing_main_effects"></a>
# ## Computing Main Effects
# 
# Now we compute the main effects of each variable using the results of the experimental design. We'll use some shorthand Pandas functions to compute these averages: the groupby function, which groups rows of a dataframe according to some condition (in this case, the value of our variable of interest $x_i$).

# In[8]:


# Compute the mean effect of the factor on the response,
# conditioned on each variable
labels = ['x1','x2','x3']

main_effects = {}
for key in labels:
    
    effects = results_df.groupby(key)['logy'].mean()

    main_effects[key] = sum( [i*effects[i] for i in [-1,1]] )

main_effects


# <a name="analyzing_main_effects"></a>
# ## Analyzing Main Effects
# 
# The main effect of a given variable (as defined by Yates 1937) is the average difference in the level of response as the input variable moves from the low to the high level. If there are other variables, the change in the level of response is averaged over all combinations of the other variables.
# 
# Now that we've computed the main effects, we can analyze the results to glean some meaningful information about our system. The first variable `x1` has a positive effect of 0.74 - this indicates that when `x1` goes from its low level to its high level, it increases the value of the response (the lieftime of the equipment). This means `x1` should be increased, if we want to make our equipment last longer. Furthermore, this effect was the largest, meaning it's the variable we should consider changing first. 
# 
# This might be the case if, for example, changing the value of the input variables were capital-intensive. A company might decide that they can only afford to change one variable, `x1`, `x2`, or `x3`. If this were the case, increasing `x1` would be the way to go.
# 
# In contrast, increasing the variables `x2` and `x3` will result in a decrease in the lifespan of our equipment (makes the response smaller), since these have a negative main effect. These variables should be kept at their lower levels, or decreased, to increase the lifespan of the equipment.

# <a name="twowayinteractions"></a>
# ## Two-Way Interactions
# 
# In addition to main effects, a factorial design will also reveal interaction effects between variables - both two-way interactions and three-way interactions. We can use the ```itertools``` library to compute the interaction effects using the results from the factorial design.
# 
# We'll use the Pandas groupby function again, grouping by two variables this time.

# In[9]:


import itertools

twoway_labels = list(itertools.combinations(labels, 2))

twoway_effects = {}
for key in twoway_labels:
    
    effects = results_df.groupby(key)['logy'].mean()
    
    twoway_effects[key] = sum([ i*j*effects[i][j]/2 for i in [-1,1] for j in [-1,1] ])

    # This somewhat hairy one-liner takes the mean of a set of sum-differences
    #twoway_effects[key] = mean([  sum([ i*effects[i][j] for i in [-1,1] ]) for j in [-1,1]  ])

twoway_effects


# This one-liner is a bit hairy:
# 
# ```
# twoway_effects[key] = sum([ i*j*effects[i][j]/2 for i in [-1,1] for j in [-1,1] ])
# ```
# 
# What this does is, computes the two-way variable effect with a multi-step calculation, but does it with a list comprehension. First, let's just look at this part:
# 
# ```
# i*j*effects[i][j]/2 for i in [-1,1] for j in [-1,1]
# ```
# 
# This computes the prefix ```i*j```, which determines if the interaction effect ```effects[i][j]``` is positive or negative. We're also looping over one additional dimension; we multiply by 1/2 for each additional dimension we loop over. These are all summed up to yield the final interaction effect for every combination of the input variables.
# 
# If we were computing three-way interaction effects, we would have a similar-looking one-liner, but with `i`, `j`, and `k`:
# 
# ```
# i*j*k*effects[i][j][k]/4 for i in [-1,1] for j in [-1,1] for k in [-1,1]
# ```

# <a name="analyzing_twowayinteractions"></a>
# ## Analyzing Two-Way Interactions
# 
# As with main effects, we can analyze the results of the interaction effects analysis to come to some useful conclusions about our physical system. A two-way interaction is a measure of how the main effect of one variable changes as the level of another variable changes. A negative two-way interaction between $x_2$ and $x_3$ means that if we increase $x_3$, the main effect of $x_2$ will be to decrase the response; or, alternatively, if we increase $x_2$, the main effect of $x_3$ will be to decrease the response.
# 
# In this case, we see that the $x_2-x_3$ interaction effect is the largest, and it is negative. This means that if we decrease both $x_2$ and $x_3$, it will increase our response - make the equipment last longer. In fact, all of the variable interactions have the same result - increasing both variables will decrease the lifetime of the equipment - which indicates that any gains in equipment lifetime accomplished by increasing $x_1$ will be nullified by increases to $x_2$ or $x_3$, since these variables will interact.
# 
# Once again, if we are limited in the changes that we can actually make to the equipment and input levels, we would want to keep $x_2$ and $x_3$ both at their low levels to keep the response variable value as high as possible.

# <a name="threewayinteractions"></a>
# ## Three-Way Interactions
# 
# Now let's comptue the three-way effects (in this case, we can only have one three-way effect, since we only have three variables). We'll start by using the `itertools` library again, to create a tuple listing the three variables whose interactions we're computing. Then we'll use the Pandas `groupby()` feature to partition each output according to its inputs, and use it to compute the three-way effects.

# In[10]:


import itertools

threeway_labels = list(itertools.combinations(labels, 3))

threeway_effects = {}
for key in threeway_labels:
    
    effects = results_df.groupby(key)['logy'].mean()
    
    threeway_effects[key] = sum([ i*j*k*effects[i][j][k]/4 for i in [-1,1] for j in [-1,1] for k in [-1,1] ])

threeway_effects


# <a name="analyzing_threewayinteractions"></a>
# ## Analysis of Three-Way Effects
# 
# While three-way interactions are relatively rare, typically smaller, and harder to interpret, a negative three-way interaction esssentially means that _increasing_ these variables, all together, will lead to interactions which lower the response (the lifespan of the equipment) by -0.082, which is equivalent to decreasing the lifespan of the equipment by one cycle. However, this effect is very weak comapred to main and interaction effects.
# 

# <a name="fitting_responsesurface"></a>
# # Fitting a Polynomial Response Surface
# 
# While identifying general trends and the effects of different input variables on a system response is useful, it's more useful to have a mathematical model for the system. The factorial design we used is designed to get us coefficients for a linear model $\hat{y}$ that is a linear function of input variables $x_i$, and that predicts the actual system response $y$:
# 
# $$
# \hat{y} = a_0 + a_1 x_1 + a_2 x_2 + a_3 x_3 + a_{12} x_1 x_2 + a_{13} x_1 x_3 + a_{23} x_2 x_3 + a_{123} x_1 x_2 x_3 
# $$
# 
# To determine these coefficients, we can obtain the effects we computed above. When we computed effects, we defined them as measuring the difference in the system response that changing a variable from -1 to +1 would have. Because this quantifies the change per two units of x, and the coefficients of a polynomial quantify the change per one unit of x, the effect must be divided by two.
# 

# In[12]:


s = "yhat = "

s += "%0.3f "%(results_df['logy'].mean())

for i,k in enumerate(main_effects.keys()):
    if(main_effects[k]<0):
        s += "%0.3f x%d "%( main_effects[k]/2.0, i+1 )
    else:
        s += "+ %0.3f x%d "%( main_effects[k]/2.0, i+1 )

print(s)


# Thus, the final result of the experimental design matrix and the 8 experiments that were run is the following polynomial for $\hat{y}$, which is a model for $y$, the system response:
# 
# $$
# \hat{y} = 2.744 - 0.295 x_1 - 0.175 x_2 + 0.375 x_3
# $$

# <a name="uncertainty"></a>
# ## The Impact of Uncertainty
# 
# The main and interaction effects give us a _more_ quantitative idea of what variables are important, yes. They can also be important for identifying where a model can be improved (if an input is linked strongly to a system response, more effort should be spent understanding the nature of the relationship). 
# 
# But there are still some practical considerations missing from the implementation above. Specifically, in the real world it is impossible to know the system repsonse, $y$, perfectly. Rather, we may measure the response with an instrument whose uncertainty has been quantified, or we may measure a quantity multiple times (or both). How do we determine the impact of that uncertainty on the model?
# 
# Ultimately, factorial designs are based on the underlying assumption that the response $y$ is a linear function of the inputs $x_i$. Thus, for the three-factor full factorial experiment design, we are collecting data and running experiments in such a way that we obtain a model $\hat{y}$ for our system response $y$, and $\hat{y}$ is a linear function of each factor:
# 
# $$
# \hat{y} = a_0 + a_1 x_1 + a_2 x_2 + a_3 x_3
# $$

# The experiment design allows us to obtain a value for each coefficient $a_0$, $a_1$, etc. that will fit $\hat{y}$ to $y$ to the best of its abilities. 
# 
# __Thus, uncertainty in the measured responses $y$ propagates into the linear model in the form of uncertainty in the coefficients $a_0$, $a_1$, etc.__

# <a name="uncertainty_example"></a>
# ## Uncertainty Quantfication: Factory Example
# 
# For example, suppose that we're dealing with a machine on a factory floor, and we're measuring the system response $y$, which is a machine failure. Now, how do we know if a machine has failed? Perhaps we can't see its internals, and it still makes noise. We might find out that a machine has failed by seeing it emit smoke. But sometimes, machines will emit smoke before they fail, while other times, machines will only smoke after they've failed. We don't know exactly how many life cycles the machines went through, but we can quantify what we know. We can measure the mean $\overline{y}$ and variance $\sigma^2$ in a controlled setting, so that when a machine starts smoking, we have a probability distribution assigning probabilities to different times of failure (i.e., there is a 5% chance it failed more than 1 hour ago).
# 
# Once we obtain the variance, or $\sigma^2$, we can obtain the value of $\sigma$, which represents the distribution of uncertainty. Assuming 2 sigma is acceptable (covers 95% of cases), we can add or subtract $\sigma$ from the estimate of parameters.

# <a name="uncertainty_numbers"></a>
# ## Uncertainty Numbers
# 
# To obtain an estimate of the uncertainty, the experimentalist will typically make several measurements at the center point, that is, where all parameter levels are 0. The more samples are taken at this condition, the better characterized the distribution of uncertainty becomes. These center point samples can be used to construct a Gaussian probability distribution function, which yeilds a variance, $\sigma^2$ (or, to be proper, an estimate $s^2$ of the real variance $\sigma^2$). This parameter is key for quantifying uncertainty.

# <a name="uncertainty_measurements"></a>
# ## Using Uncertainty Measurements 
# 
# Suppose we measure $s^2 = 0.0050$. Now what?
# 
# Now we can obtain the variance of all measurements, and the variance in the effects that we computed above. These are computed via:
# 
# $$
# Var_{mean} = V(\overline{y}) = \dfrac{\sigma^2}{2^k} \\
# Var_{effect} = \dfrac{4 \sigma^2}{2^k}
# $$

# In[14]:


sigmasquared = 0.0050
k = len(inputs_df.index)
Vmean = (sigmasquared)/(2**k)
Veffect = (4*sigmasquared)/(2**k)
print("Variance in mean: %0.6f"%(Vmean))
print("Variance in effects: %0.6f"%(Veffect))


# Alternatively, if the responses $y$ are actually averages of a given number $r$ of $y$-observations, $\overline{y}$, then the variance will shrink:
# 
# $$
# Var_{mean} = \dfrac{\sigma^2}{r 2^k} \\
# Var_{effect} = \dfrac{4 \sigma^2}{r 2^k}
# $$
# 
# The variance gives us an estimate of sigma squared, and if we have sigma squared we can obtain sigma. Sigma is the quantity that represents the range of response values that captures 1 sigma, or 66%, of the probable values of $y$ with $\hat{y}$. Adding a plus or minus sigma means we are capturing 2 sigma, or 95%, of the probable values of $y$.
# 
# Taking the square root of the variance gives $\sigma$:

# In[16]:


print(np.sqrt(Vmean))
print(np.sqrt(Veffect))


# <a name="uncertainty_accounting"></a>
# ## Accounting for Uncertainty in Model
# 
# Now we can convert the values of the effects, and the values of $\sigma$, to values for the final linear model:
# 
# $$
# \hat{y} = a_0 + a_1 x_1 + a_2 x_2 + a_3 x_3 + a_{12} x_1 x_2 + a_{13} x_1 x_3 + a_{23} x_2 x_3 + a_{123} x_1 x_2 x_3 
# $$
# 
# We begin with the case where each variable value is at its middle point (all non-constant terms are 0), and 
# 
# $$
# \hat{y} = a_0
# $$
# 
# In this case, the standard error is $\pm \sigma$ as computed for the mean (or overall) system response,
# 
# $$
# \hat{y} = a_0 \pm \sigma_{mean}
# $$
# 
# where $\sigma_{mean} = \sqrt{Var(mean)}$.
# 

# In[18]:


unc_a_0 = np.sqrt(Vmean)
print(unc_a_0)


# The final polynomial model for our system response prediction $\hat{y}$ therefore becomes:
# 
# $$
# \hat{y} = ( 2.744 \pm 0.025 ) - ( 0.295 \pm 0.025 ) x_1 - (0.175 \pm 0.025) x_2 + (0.375 \pm 0.025) x_3
# $$

# <a name="discussion"></a>
# # Discussion
# 
# At this point, we would usually dive deeper into the details of the actual problem of interest. By tying the empirical model to the system, we can draw conclusions about the physical system - for example, if we were analyzing a chemically reacting process, and we found the response to be particularly sensitive to temperature, it would indicate that the chemical reaction is sensitive to temperature, and that the reaction should be studied more deeply (in isolation from the more complicated system) to better understand the impact of temperature on the response.
# 
# It's also valuable to explore the linear model that we obtained more deeply, by looking at contours of the response surface, taking first derivatives, and optimizing the input variable values to maximize or minimize the response value. We'll leave those tasks for later, and illustrate them in later notebooks.
# 
# At this point we have accomplished the goal of illustrating the design, execution, and analysis of a two-level, three-factor full factorial experimental design, so we'll leave things at that.

# <a name="conclusion"></a>
# # Conclusion
# 
# In this notebook, we've covered a 2-level, three-factor factorial design from start to finish, including incorporation of uncertainty information. The design of the experiment was made simple by using the `itertools` and `pandas` libraries, and we showed how to transform variables to have low and high levels, as well as demonstrating a system response transformation. The results were analyzed to obtain a linear polynomial model.
# 
# However, this process was a bit cumbersome. What we'll see in later notebooks is that we can use Python modules designed for statistical modeling to fit linear models to data using least squares and regression, and carry the analysis further.

# In[ ]:




