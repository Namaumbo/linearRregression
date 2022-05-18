import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

Dataset = pd.read_csv("C:\\Users\\salaryData.csv")
Experience = Dataset[['years of experience']]
salary = Dataset['salary']
# training the data
x_train, x_test, y_train, y_test = train_test_split(Experience, salary, test_size=0.20, random_state=0)
newData = np.array([[1.2, 11, 19, 12]])
predictions = newData.reshape(-1, 1)

# trying to fit the linear regreassion model
slrModel = LinearRegression()
slrModel.fit(x_train, y_train)
yPredictionSlr = slrModel.predict(x_test)
newPrediction = slrModel.predict(predictions)

slrDifference = pd.DataFrame({'Actual value': y_test, 'predicted value': yPredictionSlr})
plt.scatter(x_test, y_test)
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.title("graph of Salary against years of experience")
plt.plot(x_test, yPredictionSlr, 'Red', label="data points")
plt.legend()
plt.show()

meanAbbErr = metrics.mean_absolute_error(y_test, yPredictionSlr)
meanSErr = metrics.mean_squared_error(y_test, yPredictionSlr)
rootMeanSquare = np.sqrt(metrics.mean_squared_error(y_test, yPredictionSlr))
print('Root Mean Square Error', rootMeanSquare)
print('Mean Square Error', meanSErr)
