import torch

# L0 test
predicted_u = torch.tensor([1, 2, 3, 4, 5], dtype=float)
ground_truth_tau = torch.tensor([4, 5, 6], dtype=float)
L0_result = L0(predicted_u, ground_truth_tau)
# print("L0 result: ", L0_result) # should be 8.9424

# L0 batch test
batch_predicted_u = torch.tensor([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]], dtype=float)
batch_ground_truth_tau = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=float)
L0_batch_result = L0_batch(batch_predicted_u, batch_ground_truth_tau)
print("L0 batch result: ", L0_batch_result) # should be 8.9424 + 182.68805 = 191.63045

# L1 batch test
pred_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
ground_truth_tensor = torch.tensor([6, 7, 8, 9, 10], dtype=torch.float)
L1_result = L1_batch(pred_tensor, ground_truth_tensor)
print("L1 batch result: ", L1_result) # Should be 25

# L2 test
u_predicted_test = torch.tensor([-40000, 20000, 90, 50000, 90])
L2_result = L2(u_predicted_test) # should be 10000
# print("L2 result: ", L2_result)

# L2 batch test
u_predicted_batch = torch.tensor([[-40000, 20000, 90, 50000, 90],
                                [-30000, 40000, 90, 50000, 90],
                                [-20000, 20000, 190, 50000, 90],
                                [-20000, 20000, 90, 70000, 90],
                                [-20000, 20000, 90, 50000, 190]])
L2_batch_result = L2_batch(u_predicted_batch) # should be 10000
print("L2 batch result: ", L2_batch_result) # should be 10k+10k+10+10k+10 = 30020

# L3 test
tensor = torch.tensor([[0, 0, 0, 0, 0],
                         [50, 50, 5, 50, 5],
                         [2000, 2000, 20, 2000, 20]])
result = L3(tensor)
print("L3 Loss: ", result) # should be 2860

# L4 test
u = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
print("L4 Loss", L4(u)) # result 202.4809

# L5

u = torch.tensor([[0, 0, -95, 0, 10],
                  [0, 0, 10, 0, -95],
                  [0, 0, 95, 0, 10],
                  [0, 0, 10, 0, 95]])

print("L5 Loss", L5(u)) # should print 4, since 4 values in 3rd and 5th column are out of range


print(u_tensor)
print(tau_tensor)
print("Combined loss: ", combined_loss(u_tensor, tau_tensor, tau_tensor))