import time

import torch
from models import *
from torch.utils.tensorboard import SummaryWriter

from losses import *


# Define combined loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau):

        return combined_loss(batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau)

# Select network architecture
model = dense_arch()

# Load data
std_tau_tensor_train = torch.load('data.pt', weights_only=False)["std_tau_tensor_train"]
print(std_tau_tensor_train.shape)

# Move model and tensor to cuda if available
if torch.cuda.is_available():
    print("Cuda available: Moving model and tensor to cuda")
    model = model.to('cuda')
    std_tau_tensor_train = std_tau_tensor_train.to('cuda')

# Check if model and input_tensor are on cuda
print("Is model on cuda: ", next(model.parameters()).is_cuda)
print("Is tensor on cuda: ", std_tau_tensor_train.is_cuda)

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
loss_fn = CustomLoss()

# Training
num_epochs = 10
batch_size = 1024

# Tensorboard
writer = SummaryWriter()

start = time.time()

for epoch in range(num_epochs):

    batch = 0

    for start_index in range(0, std_tau_tensor_train.size(0), batch_size):

      batch += 1

      end_index = start_index + batch_size

      model.train()

      tau_train = std_tau_tensor_train[start_index:end_index]
      # print(tau_train.shape)

      batch_predicted_u = model[:5](tau_train)
      batch_predicted_tau = model(tau_train)
      # print(batch_predicted_tau)

    #   print("Is tau_train on cuda: ", tau_train.is_cuda)
    #   print("Is batch_predicted_u on cuda: ", batch_predicted_u.is_cuda)
    #   print("Is batch_predicted_tau on cuda: ", batch_predicted_tau.is_cuda)

      loss = loss_fn(batch_predicted_u, batch_predicted_tau, tau_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f"Epoch [{epoch+1}/{num_epochs}], Batch: {batch}, Loss: {loss.item()}")

    writer.add_scalar("Loss/train", loss, epoch+1)

    # shuffle input tensor after each epoch to reduce bias towards end of training data
    shuffled_indices = torch.randperm(std_tau_tensor_train.size(0))
    std_tau_tensor_train = std_tau_tensor_train[shuffled_indices]

end = time.time()
print("Elapsed wall clock time: ", end - start)

# Save model for rendering
torch.save(model, "model.pt")

writer.flush()

# TensorBoard HowTo: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html