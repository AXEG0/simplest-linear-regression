from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# First column is Years of experience
# Second column is People in supervision
x_data = [[2, 1],
          [2, 0],
          [3, 4]]

# And this is the Salary
y_data = [2, 1, 4]

# Let's fit a Linear Regression model
model = LinearRegression().fit(x_data, y_data)

# Here let's check the average model error value
print("MSE:", mean_squared_error(y_data, model.predict(x_data)))

# Finally, let's predict the salary with 5 Years of experience and 8 People in supervision
prediction = model.predict([[5, 8]])
print('Predicted response:', prediction)
