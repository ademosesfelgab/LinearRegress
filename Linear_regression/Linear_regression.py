# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:05:05 2019

@author: Guru
"""
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
from sklearn.linear_model  import LinearRegression
import statsmodels.api as sm
## Here's the data from the example:
mouse = pd.DataFrame({"Correct":[17, 13, 12, 15, 16, 14, 16, 16, 18, 19],
  "Attitude":[94, 73, 59, 80, 93, 85, 66, 79, 77, 91]})
print(mouse)


#init_notebook_mode()

## plot a x/y scatter plot with the data
trace0 = Scatter(
    x=mouse.Correct,
    y=mouse.Attitude,
    mode='markers')

# create a "linear model" - that is, do the regression
X2 = sm.add_constant(mouse.iloc[:,0:1].values)
est = sm.OLS(mouse.iloc[:,1].values, X2)
est2  = est.fit()

## generate a summary of the regression
print("summary()\n",est2.summary())


"""
# create a "linear model" - that is, do the regression
lm = LinearRegression()
lm.fit(mouse.iloc[:,0:1].values,mouse.iloc[:,1].values)

# add the regression line to our x/y scatter plot
trace1 = Scatter(
    x = mouse.weight,
    y = lm.predict(mouse.iloc[:,0:1].values)
)
"""
lm = LinearRegression()
lm.fit(mouse.iloc[:,0:1].values,mouse.iloc[:,1].values)

# add the regression line to our x/y scatter plot
trace2 = Scatter(
    x = mouse.Correct,
    y = est2.predict(X2)
)


# Plot
data = [trace0,trace2]

layout = Layout(
    showlegend=True,
    height=600,
    width=600,
)

fig = dict( data=data, layout=layout )
plot(fig)  