import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

Dataset = pd.read_csv("C:\\Users\\hp\\Desktop\\assignments\\50-Startups.csv")
stateDummies = pd.get_dummies(Dataset.State)
combinedDummies = pd.concat([Dataset,stateDummies],axis='columns')
xIndependent = combinedDummies.drop(['State'],axis='columns')
yIndependent = combinedDummies.Profit