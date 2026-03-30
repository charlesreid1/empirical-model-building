#!/usr/bin/env python
# coding: utf-8

# # Fractionating $2^k$ Factorial Designs
# 
# 
# ## Motivation
# 
# The prior section showed an example of what an experimental design might like look like for 6 variables. However, this resulted in a $2^6 = 64$ experiment design campaign. This is potentially a major issue - if my experiments take 6 hours, and have to be staggered over working hours on weekdays, you're looking at almost 90 days turnaround time, assuming each experiment is carried out flawlessly. This is simply not a realistic view of experimentation.
# 
# In addition, we saw that a five-coefficient model captured nearly as much detail as a 64-coefficient model. By reducing the number of input variables we looked at, we turned certain experiments into replicates (because the only thing changed bewteen them were insignificant variables or variable combinations). 
# 
# But we can halve or quarter our effort, and substantially improve our effectiveness in the lab, by carefully selecting experiments at each stage of the experiment to reveal a maximum amount of information, and avoiding as much as possible these kinds of duplicate experiments, through a fractional factorial design.

# In[1]:


import pandas as pd
import itertools
import numpy as np
import seaborn as sns
import pylab

import scipy.stats as stats
import statsmodels.api as sm


# After re-casting the problem in a general form, we begin with the experimental design matrix. If we were to construct the full factorial for our $2^6$ factorial example, we would again have 64 rows in our experimental design matrix dataframe, corresponding to 64 experiments to run.

# In[2]:


column_labs = ['x%d'%(i+1) for i in range(6)]
encoded_inputs = list( itertools.product([-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]) )
doe = pd.DataFrame(encoded_inputs,columns=column_labs)
print(len(doe))


# ## Design Matrix
# 
# Let's talk a bit more about the design matrix. Each column of the design matrix corresponds to a unique coded input variable value $(-1,+1)$. But each experiment also has a corresponding coded value for each two-variable interaction $x_i,x_j$, and for each three-variable interaction $x_k,x_m,x_n$, and so on. 
# 
# These interactions are simply the product of each coded variable value. For example, if 
# 
# $$
# x_1 = -1 \\
# x_2 = +1 \\
# x_3 = +1
# $$
# 
# then two-variable interaction effects can be computed as:
# 
# $$
# x_{12} = -1 \times +1 = -1 \\
# x_{13} = -1 \times +1 = -1 \\
# x_{23} = +1 \times +1 = +1 \\
# $$
# 
# and three-variable interaction effects are:
# 
# $$
# x_{123} = -1 \times -1 \times +1 = +1
# $$
# 
# Now we can add new columns to our experimental design matrix dataframe, representing coded values for higher-order interaction effects:

# In[3]:


doe['x1-x2-x3-x4'] = doe.apply( lambda z : z['x1']*z['x2']*z['x3']*z['x4'] , axis=1)
doe['x4-x5-x6']    = doe.apply( lambda z : z['x4']*z['x5']*z['x6'] , axis=1)
doe['x2-x4-x5']    = doe.apply( lambda z : z['x2']*z['x4']*z['x5'] , axis=1)

doe[0:10]


# The multi-variable columns can be used to fractionate our design.
# 
# ## Half Factorial
# 
# Suppose we pick a high-order interaction effect at random - e.g., $x_1 \times x_2 \times x_3 \times x_4$ - and assume it will be unimportant. Our assumption allows us to cut out any experiments that are intended to give us information about the effect of $x_1 x_2 x_3 x_4$. 
# 
# For any two groups of experiments, if one group has
# 
# $$x_1 x_2 x_3 x_4 = +1$$ 
# 
# and the other group has
# 
# $$x_1 x_2 x_3 x_4 = -1$$
# 
# then based on our assumption that that interaction effect will be unimportant, one of those two groups can be thrown out. 
# 
# Fortuitously, the first time a variable is eliminated, no matter which variable it is, the number of experiments is cut in half. Further eliminations of variables continue to cut the number of experiments in half. So a six-factor experimental design could be whittled down as follows:
# 
# Six-factor, two-level experiment design:
# * $n=2$, $k=6$, $2^6$ experimental design
# * Full factorial: $2^6 = 64$ experiments
# * Half factorial: $2^{6-1} = 32$ experiments
# * $\frac{1}{4}$ Fractional factorial: $2^{6-2} = 16$ experiments
# * $\frac{1}{8}$ Fractional factorial: $2^{6-3} = 8$ experiments
# * $\frac{1}{16}$ Fractional factorial: $2^{6-4} = 4$ experiments
# 
# In general, for an $n^k$ experiment design ($n$ factor, $k$ level), a $\dfrac{1}{2^p}$ fractional factorial can be defined as:
# 
# * $\dfrac{1}{2^p}$ Fractional factorial: $2^{n-p}$ experiments
# 
# Note that as the fractional factorial gets narrower, and the experiments get fewer, the number of aliased interaction effects gets larger, until not even interaction effects can be distinguished, but only main effects. (Screening designs, such as Plackett-Burman designs, are based on this idea of highly-fractionated experiment design; we'll get into that later.)
# 
# For now, let's look at the half factorial: 32 experiments, with the reduction in varaibles coming from aliasing the interaction effect $x_1 x_2 x_3 x_4$:

# In[5]:


print(len( doe[doe['x1-x2-x3-x4']==1] ))


# ## Costs and Benefits
# 
# The benefits are obvious - we've halved the number of experiments our experiment design requires. But at what cost?
# 
# The first 32 experiments, where $x_1 x_2 x_3 x_4 = +1$, give us information at a positive level of that input variable combination. To get information at a negative level of that input variable combination (i.e., $x_1 x_2 x_3 x_4 = -1$), we need 32 additional experiments.
# 
# Our assumption is that changing $x_1 x_2 x_3 x_4$ from high to low will have no effect on the observable $y$.
# 
# This also modifies the information we get about higher-order interaction effects. For example, we've assumed:
# 
# $$
# x_1 x_2 x_3 x_4 = +1
# $$
# 
# We can use this identity to figure out what information we're missing when we cut out the 32 experiments. Our assumption about the fourth-order interaction also changes fifth- and sixth-order interactions:
# 
# $$
# (x_1 x_2 x_3 x_4) = (+1) \\
# (x_1 x_2 x_3 x_4) x_5 = (+1) x_5 \\
# x_1 x_2 x_3 x_4 x_5 = x_5
# $$
# 
# meaning the fifth-order interaction effect $x_1 x_2 x_3 x_4 x_5$ has been aliased with the first-order main effect $x_5$. This is a safe assumption since it is extremely unlikely that a fifth-order interaction effect could be confounded with a first-order main effect. We can derive other relations, using the fact that any factor squared is equivalent to $(+1)$, so that:
# 
# $$
# (x_1 x_2 x_3 x_4) = +1 \\
# (x_1 x_2 x_3 x_4) x_1 = (+1) x_1 \\
# (x_1^2 x_2 x_3 x_4) = (+1) x_1 \\
# x_2 x_3 x_4 = x_1 
# $$
# 
# 

# -----
# 
# The sequence of variables selected as the interaction effect to be used as the experimental design basis is called the generator, and is denoted $I$:
# 
# $$
# I = x_1 x_2 x_3 x_4
# $$
# 
# and we set $I=+1$ or $I=-1$.

# In[6]:


# Defining multiple DOE matrices:

# DOE 1 based on identity I = x1 x2 x3 x4 
doe1 = doe[doe['x1-x2-x3-x4']==1]

# DOE 2 based on identity I = x4 x5 x6
doe2 = doe[doe['x4-x5-x6']==-1]

# DOE 3 based on identity I = x2 x4 x5
doe3 = doe[doe['x2-x4-x5']==-1]


# In[7]:


doe1[column_labs].T


# In[8]:


doe2[column_labs].T


# In[9]:


doe3[column_labs].T


# Each of the dataframes above represents a different fractional factorial design. 
# 

# ## $\frac{1}{4}$ Fractional Designs
# 
# 
# To further reduce the number of experiments, two identities can be used. The number of experiments is cut in half for each identity. We already have one identity,
# 
# $$
# I = x_1 x_2 x_3 x_4 = 1
# $$
# 
# now let's define another one:
# 
# $$
# I_2 = x_4 x_5 x_6 = 1
# $$
# 
# Our resulting factorial matrix can be reduced the same way. In Python, we use the ```logical_and``` function to ensure our two conditions are satisfied.

# In[10]:


quarter_fractional_doe = doe[ np.logical_and( doe['x1-x2-x3-x4']==1, doe['x4-x5-x6']==1 ) ]
print("Number of experiments: %d"%(len(quarter_fractional_doe[column_labs])))
quarter_fractional_doe[column_labs].T


# With the quarter-fractional factorial design, what information do we lose? We know already which interaction effects are aliased with main effects:
# 
# $$
# x_4 x_5 x_6 = (+1) \\
# (x_4 x_5 x_6) x_1 = (+1) x_1 \\
# x_1 x_4 x_5 x_6 = x_1 \\
# x_2 x_4 x_5 x_6 = x_2 \\
# x_3 x_4 x_5 x_6 = x_3
# $$
# 
# We can use this information to design our experiments to cover particular interaction effects we know to be important, or ignore others we don't expect to be significant.
