# Helper for producing a model

from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import os

# go to desired directory
os.chdir(os.path.join(os.path.expanduser('~'), 'optiguardml/models'))

### build dummy scikit model with 1 input and one output
X = np.array([[1], [2], [3], [4]])
y = np.array([[1], [2], [3], [4]])

reg = LinearRegression().fit(X, y)

print ("regression score", reg.score(X, y))
print("regression coefficients", reg.coef_)
print ("regression intercept", reg.intercept_)

# save model to disk
pickle.dump(reg, open('model.ml', 'wb'))

print('model saved')
