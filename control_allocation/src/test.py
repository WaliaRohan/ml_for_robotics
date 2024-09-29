import torch
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

from losses import *
# import torch.nn as nn
from models import *


# Define combined loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau):

        return combined_loss(batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau)

# Select network architecture
model = dense_arch()

# Save model for rendering
torch.save(model, "model.pt")

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
loss_fn = CustomLoss()

# Testing

std_tau_tensor_test = torch.load('data.pt', weights_only=False)["std_tau_tensor_test"]
print(std_tau_tensor_test.shape)

num_epochs = 10
batch_size = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Tensorboard
writer = SummaryWriter()

for epoch in range(num_epochs):

    batch = 0

    for start_index in range(0, std_tau_tensor_test.size(0), batch_size):

      batch += 1

      end_index = start_index + batch_size

      model.eval()

      tau_test = std_tau_tensor_test[start_index:end_index]
      # print(tau_test.shape)

      batch_predicted_u = model[:5](tau_test)
      batch_predicted_tau = model(tau_test)
      # print(batch_predicted_tau)

      loss = loss_fn(batch_predicted_u, batch_predicted_tau, tau_test)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f"Epoch [{epoch+1}/{num_epochs}], Batch: {batch}, Loss: {loss.item()}")

      writer.add_scalar("Loss/test", loss, epoch)

    # shuffle input tensor after each epoch to reduce bias towards end of testing data
    shuffled_indices = torch.randperm(std_tau_tensor_test.size(0))
    std_tau_tensor_test = std_tau_tensor_test[shuffled_indices]

writer.flush()

# TensorBoard HowTo: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html