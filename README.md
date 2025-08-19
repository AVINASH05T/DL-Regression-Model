# Developing a Neural Network Regression Model
### NAME : AVINASH T
### REG NO : 212223230026
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: AVINASH T

### Register Number: 212223230026

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(71)
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

plt.scatter(X, y, color="black")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Genearted data for linear Regression')
plt.show()
```

# Initialize the Model, Loss Function, and Optimizer

```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #Include your code here
    def forward(self, x):
    return self.linear(x)

torch.manual_seed(59)
model = Model(1, 1)

initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: AVINASH T")
print("Registor No: 212223230026")
print(f"Initial Weight: {initial_weight:.8f} , Initial Bias: {initial_bias:.8f}")

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 100
losses = []

for epoch in range(1, epochs+1):
  optimizer.zero_grad()
  y_pred = model(X)
  loss = loss_function(y_pred, y)
  losses.append(loss.item())

  loss.backward()
  optimizer.step()

  print(f'Epoch: {epoch:2} loss: {loss.item():10.8f}, '
        f'weight: {model.linear.weight.item():10.8f}, '
        f'bias: {model.linear.bias.item():10.8f}')
plt.plot(range(epochs), losses, color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.show()

final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName: AVINASH T")
print("Registor No: 212223230026")
print(f"\nFinal Weight: {final_weight:.8f} , Final Bias: {final_bias:.8f}")

x1 = torch.tensor([X.min().item(), X.max().item()])
y1 = x1 * final_weight + final_bias

plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trained Model: Best-Fit Line")
plt.legend()
plt.show()

x_new = torch.tensor([120.0])
y_new_pred = model(x_new).item()
print("\nName: AVINASH T")
print("Registor No: 212223230026")
print(f"Predicted for X = 120: {y_new_pred:.8f}")

```

### Dataset Information

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/5483891b-e13f-41c2-a22e-39e771861f99" />

<img width="846" height="108" alt="image" src="https://github.com/user-attachments/assets/d1856c5c-c96c-49e6-b16a-af1379303a38" />


### OUTPUT
Training Loss Vs Iteration Plot
Best Fit line plot

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/ba9548e5-86ed-41aa-8e5f-1be9fc8fd974" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/cebe45aa-b54e-446a-8ffb-d068ede6ffbc" />

<img width="700" height="103" alt="image" src="https://github.com/user-attachments/assets/da8b7de1-ba79-4f10-9fd3-ea1977e410c6" />


### New Sample Data Prediction:

<img width="650" height="44" alt="image" src="https://github.com/user-attachments/assets/951b9508-058d-46f1-8b98-b7cc6de35eb5" />
<img width="731" height="25" alt="image" src="https://github.com/user-attachments/assets/cdccad05-cfb7-4678-9905-66f88c35d274" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
