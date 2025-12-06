#1) Design Model (input, output size, forward pass)
#2) Construct loss and optimizer
#3) Training loop
# - Forward pass: computer prediction
# - Backward pass: gradients
# - Update weights
import torch
import torch.nn as nn


#f = w * x
#f = 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features
#model = nn.Linear(input_size, output_size)

# gradient
# function J = 1/N * (wx-y)^2
# dJ/dw = 1/N * 2x(wx- y)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegressionModel(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss() #callable function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    l.backward() # dl/dw

    #update weights
    optimizer.step()

    # zero the gradient
    optimizer.zero_grad()

    if epoch % 2 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
