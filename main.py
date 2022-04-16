# First column is Years of experience
# Second column is People in supervision
X = [[2, 1],
     [2, 0],
     [3, 4]]

# And this is the Salary
y = [2000,
     1000,
     4000]

# Let's fit a Linear Regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)

# Finally, let's predict the salary with 5 Years of experience and 8 People in supervision
prediction = model.predict([[5, 8]])
print('Prediction:', round(*prediction))

# Here let's check the average model error value
from sklearn.metrics import mean_absolute_error

mae = round(mean_absolute_error(y, model.predict(X)), 10)
print("Average error:", mae)
