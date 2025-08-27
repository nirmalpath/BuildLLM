# Notes
# Neural Network: weighted sum / bias: offset making it easier or harder for neuron to fire / activation: final o/p
# Layers in feed forward network: i/p / hidden: connected to all o/p from previous layer and bias vector (128) / o/p: neurons depends on task
# Activation Functions: to learn non-linear patterns / sigmoid: f(x) = 1 / (1 + e^-x); has vanishing gradient problem; (0,1) /
#                       Tanh: f(x) = (e^x-e^-x)/(e^x+e^-x); (0,1) / ReLU: f(x) = max(0,x);  efficient / softmax: multi-class problem
# Training Loop: Init with random numbers / forward pass: input @ weights + bias; followed by activation / loss calc: cross entripy for class n MSE for regression /
#                backward pass: gradient of loss wrt all weights n bias / update weights: optimizer / new wt = old wt - (learning rate * gradient) /
#                repeat
# Adam: Adaptive Moment Estimation / beyond gradient descent / momentum: velocity of gradients n adaptive learning rates: separate rates

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class SimpleFFN(nn.Module):
    def __init__(self):
        super(SimpleFFN, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate synthetic data: y = 2x + 1
x_train = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)
y_train = 2 * x_train + 1

# Model, loss, and optimizer
model = SimpleFFN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/300], Loss: {loss.item():.4f}")

# Test on a sample input
model.eval()
x_test = torch.tensor([[5.0]])
y_pred = model(x_test)
print(f"Prediction for x=5: {y_pred.item():.2f} (Expected: 11.0)")