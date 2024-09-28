import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from losses import *


# Define combined loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau):

        return combined_loss(batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau)


# Create Neural Network
model = nn.Sequential()

# Encoder layers
model.add_module("encoder_input", nn.Linear(3, 15))
model.add_module("en_act_1", nn.ReLU())
model.add_module("encoder_hidden", nn.Linear(15, 10))
model.add_module("en_act_2", nn.ReLU())
model.add_module("encoder_output", nn.Linear(10, 5))
# output of en_act_3 = batch_predicted_u

# Decoder layers
model.add_module("decoder_input", nn.Linear(5, 25))
model.add_module("dec_act_1", nn.ReLU())
model.add_module("decoder_hidden", nn.Linear(25, 15))
model.add_module("dec_act_2", nn.ReLU())
model.add_module("decoder_output", nn.Linear(15, 3))
# output of dec_act_3 = batch_predicted_tau

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss_fn = CustomLoss()

# Training

std_tau_tensor_train = torch.load('data.pt', weights_only=False)["std_tau_tensor_train"]
print(std_tau_tensor_train.shape)

num_epochs = 1
batch_size = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Tensorboard
writer = SummaryWriter()
# https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html


for epoch in range(num_epochs):

    batch = 0

    for start_index in range(0, std_tau_tensor_train.size(0), batch_size):

      batch += 1

      end_index = start_index + batch_size

      model.train()

      Tau_train = std_tau_tensor_train[start_index:end_index]
      # print(Tau_train.shape)

      batch_predicted_u = model[:5](Tau_train)
      batch_predicted_tau = model(Tau_train)
      # print(batch_predicted_tau)


      loss = loss_fn(batch_predicted_u, batch_predicted_tau, Tau_train)
      writer.add_scalar("Loss/train", loss, batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f"Epoch [{epoch+1}/{num_epochs}], Batch: {batch}, Loss: {loss.item()}")

writer.flush()

# TensorBoard HowTo: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html