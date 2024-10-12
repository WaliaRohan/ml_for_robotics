import time

import torch
from losses import *
from models import *
from torch.utils.tensorboard import SummaryWriter


# Define combined loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau):

        return combined_loss(batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau)

# Load model and data
model = torch.load("model.pt", weights_only=False)
std_tau_tensor_test = torch.load('data.pt', weights_only=False)["std_tau_tensor_test"]
print(std_tau_tensor_test.shape)

# Move model and tensor to cuda if available
if torch.cuda.is_available():
    print("Cuda available: Moving model and tensor to cuda")
    model.to('cuda')
    std_tau_tensor_test = std_tau_tensor_test.to('cuda')

loss_fn = CustomLoss()

# Testing
num_epochs = 10
batch_size = 1024
model.eval()

# Tensorboard
writer = SummaryWriter()

start = time.time()

for epoch in range(num_epochs):

    batch = 0
    batch_loss = 0

    for start_index in range(0, std_tau_tensor_test.size(0), batch_size):

      batch += 1

      # Make a Forward Pass and get the output.
      end_index = start_index + batch_size
      tau_test = std_tau_tensor_test[start_index:end_index]
      batch_predicted_u = model[:5](tau_test)
      batch_predicted_tau = model(tau_test)

      # Record and report loss
      loss = loss_fn(batch_predicted_u, batch_predicted_tau, tau_test)
      batch_loss += loss.item()

      print(f"Epoch [{epoch+1}/{num_epochs}], Batch: {batch}, Loss: {loss.item()}")

    writer.add_scalar("Loss/test", batch_loss/batch, epoch+1)

end = time.time()
print("Elapsed wall clock time: ", end - start)

writer.flush()

# TensorBoard HowTo: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html