import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

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

  if(vector.is_cuda):
    B = B.to('cuda')

#   print("Is vector on cuda: ", vector.is_cuda)
#   print("Is B on cuda: ", B.is_cuda)

  return B

def tau(vector):

  force_vector = vector[[0, 1, 3]].unsqueeze(1)

  tau = torch.matmul(B(vector), force_vector)

  return tau

def main(n=10000, train_split=0.8):
  u_tensor = generate_random_tensor(n)

  tau_tensor = torch.zeros((n, 3))

  for (i, vector) in enumerate(u_tensor):
      if i%1000 is 0:
        print(i)
      tau_tensor[i] = tau(vector).squeeze()


  # Standardize input data
  mean = tau_tensor.mean(dim=0)
  std = tau_tensor.std(dim=0)

  # Perform standardization: (input - mean) / std
  std_tau_tensor = (tau_tensor - mean) / std

  std_tau_tensor_train,std_tau_tensor_test = train_test_split(std_tau_tensor, train_size = train_split)

  torch.save({"std_tau_tensor_train": std_tau_tensor_train, "std_tau_tensor_test": std_tau_tensor_test}, "data.pt")

  print("Training data size: ", std_tau_tensor_train.shape, "\n Testing data size: ", std_tau_tensor_test.shape)


if __name__ == "__main__":
    # Set default values
    default_n = 10000
    default_train_split = 0.8

    try:
        # Check if the user provided an argument for n
        if len(sys.argv) > 1:
            n = int(sys.argv[1])  # Convert the first argument to an integer
        else:
            n = default_n  # Use default value for n
        
        # Check if the user provided an argument for train_split
        if len(sys.argv) > 2:
            train_split = float(sys.argv[2])  # Convert the second argument to a float
            if not (0 <= train_split <= 1):
                raise ValueError("train_split should be a float between 0 and 1.")
        else:
            train_split = default_train_split  # Use default value for train_split

        # Call the main function with either user-provided or default values
        main(n, train_split)

    except ValueError:
        print("Please provide a valid integer for 'n' and a valid float for 'train_split'.")



