"""
This is a code implementing a simple, single feature Linear Regression with Stochastic Gradient Descent from scratch using only vanilla Python
"""

class StochasticLinearRegression:
	def __init__(self, learning_rate, num_epochs) -> None:
		self.weight = 0.0
		self.bias = 0.0
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs

	# Linear Regression functions
	def predict(self, x):
		return self.weight * x + self.bias  # y = wx + b, line formula

	def update_params(self, x, error):
		# Calculate the Gradient using SGD and squared error loss function
		dldw = -1 * error * x  # dldw = -2 * xi * (y-ypred), but the -2 is constant meaning it can be abstracted, leaving only the negative sign and y-ypred = error
		dldb = -1 * error  # dldb = -2 * (y-ypred), but same thing as before

		# Update parameters using gradient descent
		self.weight = self.weight - self.learning_rate * dldw
		self.bias = self.bias - self.learning_rate * dldb

	def fit(self, inputs, outputs, print_epoch=True):
		for epoch in range(self.num_epochs):
			ase = 0

			for x, y in zip(inputs, outputs):
				# Make prediction
				ypred = self.predict(x)

				# Calculate error
				error = y-ypred

				# Update the total error for the epoch
				ase += error ** 2

				# Update Parameters
				self.update_params(x, error)

			# Calculate MSE
			mse = ase / len(inputs)

			if print_epoch:
				# Print epoch results
				print(f"Epoch: {epoch} - Weight: {self.weight} - Bias: {self.bias} - MSE: {mse}")


# Creating the dataset
inputs = [1, 2, 3, 4, 5]
outputs = [26, 29, 32, 35, 38]  # 3x + 23, a randomly chosen linear function

# Creating the model
model = StochasticLinearRegression(learning_rate=0.01, num_epochs=4000)

# Fitting the model to our dataset
model.fit(inputs, outputs)

# Print end results
print(f"\nFinal Results\nWeight: {model.weight} - Bias: {model.bias}")