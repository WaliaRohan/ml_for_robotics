import numpy as np
import torch

torch.manual_seed(42)
torch.set_printoptions(sci_mode=False)

def generate_random_tensor(n):
    # Define the ranges
    F1_range = torch.tensor([-10000, 10000])
    F2_range = torch.tensor([-5000, 5000])
    alpha2_range = torch.tensor([-180, 180])
    F3_range = torch.tensor([-5000, 5000])
    alpha3_range = torch.tensor([-180, 180])

    # Generate random values within each range
    F1 = F1_range[0] + (F1_range[1] - F1_range[0]) * torch.rand(n, 1)
    F2 = F2_range[0] + (F2_range[1] - F2_range[0]) * torch.rand(n, 1)
    alpha2 = alpha2_range[0] + (alpha2_range[1] - alpha2_range[0]) * torch.rand(n, 1)
    F3 = F3_range[0] + (F3_range[1] - F3_range[0]) * torch.rand(n, 1)
    alpha3 = alpha3_range[0] + (alpha3_range[1] - alpha3_range[0]) * torch.rand(n, 1)

    # Stack them to form a 5xn tensor
    random_tensor = torch.cat([F1, F2, alpha2, F3, alpha3], dim=1)

    return random_tensor

def B(vector):

  f1 = vector[0]
  f2 = vector[1]
  a2 = torch.deg2rad(vector[2])
  f3 = vector[3]
  a3 = torch.deg2rad(vector[4])

  # Dimensions from thruster layout of vessel (paper fig 2(b))
  l1 = -14
  l2 = 14.5
  l3 = -2.7
  l4 = 2.7

  B = torch.tensor([[0, torch.cos(a2), torch.cos(a3)],
                      [1, torch.sin(a2), torch.sin(a3)],
                      [l2, l1*torch.sin(a2)  - l3*torch.cos(a2), l1*torch.sin(a3) - l4*torch.cos(a3)]])

  return B

def tau(vector):

  force_vector = vector[[0, 1, 3]].unsqueeze(1)

  tau = torch.matmul(B(vector), force_vector)

  return tau


# Generate input data

n = 4
u_tensor = generate_random_tensor(n)

tau_tensor = torch.zeros((n, 3))

for (i, vector) in enumerate(u_tensor):
    tau_tensor[i] = tau(vector).squeeze()


print(u_tensor.shape)
print(tau_tensor.shape)

torch.save({"u_tensor": u_tensor, "tau_tensor": tau_tensor}, "input_data.pt")
