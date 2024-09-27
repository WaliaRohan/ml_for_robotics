import torch.nn as nn
from sklearn.model_selection import train_test_split


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau):

        return combined_loss(batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau)

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
optimizer.zero_grad()  # Zero the parameter gradients
loss_fn = CustomLoss()

# Training

num_epochs = 1
batch_size = 10

device = "cuda" if torch.cuda.is_available() else "cpu"


for epoch in range(num_epochs):

    for start_index in range(0, tau_tensor.size(0), batch_size):

      end_index = start_index + batch_size

      model.train()

      Tau_train = tau_tensor[start_index:end_index]
      # print(Tau_train.shape)

      batch_predicted_u = model[:5](Tau_train)
      batch_predicted_tau = model(Tau_train)
      print(batch_predicted_tau)


      loss = loss_fn(batch_predicted_u, batch_predicted_tau, Tau_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(inputs)
#     loss = loss_fn(outputs, targets)

#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
