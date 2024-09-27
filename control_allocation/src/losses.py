import torch
import torch.nn as nn


# OLEG
def L0(predicted_u, ground_truth_tau):

    y = ground_truth_tau
    y_hat_cmd = tau(predicted_u).squeeze()

    # print("Inside L0")
    # print("y: ", y)
    # print("y_hat: ", y_hat_cmd)

    loss_fn = nn.MSELoss()  # Mean Square Error
    loss = loss_fn(y_hat_cmd, y)

    return loss

# OLEG
def L0_batch(batch_predicted_u, batch_ground_truth_tau):

    total_loss = 0

    sample_size = batch_ground_truth_tau.shape[0]

    for predicted_u, ground_truth_tau in zip(batch_predicted_u, batch_ground_truth_tau):
      total_loss += L0(predicted_u, ground_truth_tau)

    return total_loss

# OLEG
def L1_batch(batch_predicted_tau, batch_ground_truth_tau):

    loss_fn = nn.MSELoss()  # Mean Square Error
    loss = loss_fn(batch_predicted_tau, batch_ground_truth_tau)

    return loss


def L2(predicted_u):

    u_max = torch.tensor([30000, 30000, 180, 60000, 180]) # [f1 f2 a2 f3 a3]  Switch a2 and f3 values
    predicted_u_max = torch.abs(predicted_u)
    elementwise_max = torch.max(predicted_u_max - u_max, torch.tensor(0))

    loss = torch.sum(elementwise_max)

    return loss

def L2_batch(batch_predicted_u): #how are we sequencing this?

    loss = 0

    for predicted_u in batch_predicted_u:
      loss += L2(predicted_u)

    return loss


def L3(batch_predicted_u):
    # https://neptune.ai/blog/pytorch-loss-functions

    max_rates = torch.tensor([1000, 1000, 10, 1000, 10]) # d_f1 d_f2 d_a3 d_f3 d_a3 -> paper table 1

    delta_batch_predicted_u = torch.diff(batch_predicted_u, dim=0)

    exceeding_values = torch.abs(delta_batch_predicted_u) - max_rates

    # Compare the tensor with the thresholds (broadcasting is applied)
    mask = exceeding_values > 0

    total_loss = exceeding_values[mask].sum()

    return total_loss

def L4(batch_predicted_u):
    power = 3/2

    # Apply the power operation to columns 1, 2, and 4 (index 0, 1, and 3 in zero-indexing)
    batch_predicted_u = torch.abs(batch_predicted_u[:, [0, 1, 3]]) ** power

    return batch_predicted_u.sum()

def L5(u):

    # input u is assumed to be batch input

    forbidden_sector_lower = [-100, -80]
    forbidden_sector_upper = [80, 100]

    mask_a_1_l = (u[:, 2] > forbidden_sector_lower[0]) & (u[:, 2] < forbidden_sector_lower[1])
    mask_a_2_l = (u[:, 4] > forbidden_sector_lower[0]) & (u[:, 4] < forbidden_sector_lower[1])

    mask_a_1_u = (u[:, 2] > forbidden_sector_upper[0]) & (u[:, 2] < forbidden_sector_upper[1])
    mask_a_2_u = (u[:, 4] > forbidden_sector_upper[0]) & (u[:, 4] < forbidden_sector_upper[1])

    print(mask_a_1_l)
    print(mask_a_2_l)
    print(mask_a_1_u)
    print(mask_a_2_u)

    L5_lower = sum(mask_a_1_l) + sum(mask_a_2_l)
    L5_upper = sum(mask_a_1_u) + sum(mask_a_2_u)


    L5 = L5_lower + L5_upper

    return L5

def combined_loss(batch_predicted_u, batch_predicted_tau, batch_ground_truth_tau):

    k0 = 10**0
    k1 = 10**0
    k2 = 10**-1
    k3 = 10**-7
    k4 = 10**-7
    k5 = 10**-1

    # Define the individual terms (loss effects)
    loss_effect0 = k0 * L0_batch(batch_predicted_u, batch_ground_truth_tau)
    loss_effect1 = k1 * L1_batch(batch_predicted_tau, batch_ground_truth_tau)
    loss_effect2 = k2 * L2_batch(batch_predicted_u)
    loss_effect3 = k3 * L3(batch_predicted_u)
    loss_effect4 = k4 * L4(batch_predicted_u)
    loss_effect5 = k5 * L5(batch_predicted_u)

    # Print each loss effect separately
    # print(f"Loss effect 0: {loss_effect0}")
    # print(f"Loss effect 1: {loss_effect1}")
    # print(f"Loss effect 2: {loss_effect2}")
    # print(f"Loss effect 3: {loss_effect3}")
    # print(f"Loss effect 4: {loss_effect4}")
    # # print(f"Loss effect 5: {loss_effect5}")

    combined_loss = (loss_effect0 + loss_effect1 + loss_effect2 +
                     loss_effect3 + loss_effect4 + loss_effect5)

    return combined_loss