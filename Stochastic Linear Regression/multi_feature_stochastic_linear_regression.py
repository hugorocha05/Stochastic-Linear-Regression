"""
This is a code implementing a simple, multi feature Linear Regression with Stochastic Gradient Descent from scratch using only vanilla Python
"""

class StochasticLinearRegression:
	def __init__(self, learning_rate, num_epochs) -> None:
		self.weights = None
		self.bias = None
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs

	def init_params(self, num_features):
		self.weights = [0.0 for _ in range(num_features)]
		self.bias = 0.0

	# Linear Regression functions
	def predict(self, X):
		if self.weights != None and self.bias != None:
			weighted_sum = 0

			for x, w in zip(X, self.weights):
				weighted_sum += x * w

			return weighted_sum + self.bias  # y = wx + b, line formula
		
		else:
			raise TypeError("Weights and bias not initialized")

	def update_params(self, X, error):
		if self.weights is not None and self.bias is not None:

			# Calculate the Gradient using SGD and squared error loss function
			for i in range(len(self.weights)):
				dldw = -1 * X[i] * error  # dldw = -2 * xi * (y-ypred), but the -2 is constant meaning it can be abstracted, leaving only the negative sign and y-ypred = error
				self.weights[i] = self.weights[i] - self.learning_rate * dldw

			dldb = -1 * error  # dldb = -2 * (y-ypred), but same thing as before
			self.bias = self.bias - self.learning_rate * dldb

		else:
			raise TypeError("Weights and bias not initialized")

	def fit(self, inputs, outputs, print_epoch=True):
		if self.weights is None:
			self.init_params(len(inputs[0]))

		for epoch in range(self.num_epochs):
			ase = 0

			for X, y in zip(inputs, outputs):
				# Make prediction
				ypred = self.predict(X)

				# Calculate error
				error = y-ypred

				# Update the total error for the epoch
				ase += error ** 2

				# Update Parameters
				self.update_params(X, error)

			# Calculate MSE
			mse = ase / len(inputs)

			if print_epoch:
				# Print epoch results
				print(f"Epoch: {epoch} - Weights: {self.weights} - Bias: {self.bias} - MSE: {mse}")

# ---------------------------------------------------------------------------------------------------------------------

# Creating the dataset
inputs = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20]]

outputs = [11.9, 17.3, 16.4, 23.2, 21.7, 24.2, 31.0, 28.7, 35.2, 33.4, 34.5, 45.3, 45.8, 40.7, 46.1, 53.4, 47.9, 49.3, 62.5, 58.8]  # Approx. 2.5x + 10

"""inputs = [[1, 7], [2, 3], [5, 1], [8, 4], [10, 2]]
outputs = [46, 36, 36, 51, 49]  # 2*x1 + 3*x2 + 23, a randomly chosen linear function"""

"""inputs = [[1], [2], [3], [4], [5]]
outputs = [26, 29, 32, 35, 38]  # 3x + 23, a randomly chosen linear function"""

# Creating the model
model = StochasticLinearRegression(learning_rate=0.0005, num_epochs=30000)

# Fitting the model to our dataset
model.fit(inputs, outputs)

# Print end results
print(f"\nFinal Results\nWeights: {model.weights} - Bias: {model.bias}")

# ---------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Convert input vectors into scalar x values
x_values = [x[0] for x in inputs]

# Scatter plot of original data
plt.scatter(x_values, outputs)

# Generate regression line predictions
line_predictions = []

for x in x_values:
    y_pred = model.predict([x])
    line_predictions.append(y_pred)

# Plot regression line
plt.plot(x_values, line_predictions)

# Labels
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")

# Show plot
plt.show()