import torch


def addTensor(tensor):
    new_tensor = tensor+2
    return tensor


tensor = torch.randn(10, 3)

print("Is tensor on cuda: ", tensor.is_cuda)

if torch.cuda.is_available():
    print("Cuda available")
    tensor = tensor.to('cuda')

tensor_copy = tensor

print("Is tensor on cuda: ", tensor.is_cuda)
print("Is tensor copy on cuda: ", tensor_copy.is_cuda)

tensor_copy_2 = tensor[2:5]
print("Is tensor copy on cuda: ", tensor_copy_2.is_cuda)

shuffled_indices = torch.randperm(tensor.size(0))
tensor_copy_3 = tensor[shuffled_indices]
print("Is tensor copy on cuda: ", tensor_copy_3.is_cuda)

tensor_copy_4 = addTensor(tensor)
print("Is tensor copy on cuda: ", tensor_copy_4.is_cuda)