import torch
import torch.nn.functional as F
import numpy as np


# tensor_rows = []

# def split_tensor(tensor_1, tensor_2):
#     result = []
#     start_index = 0

#     for length in tensor_2:
#         end_index = start_index + length
#         result.append(tensor_1[start_index:end_index])
#         start_index = end_index

#     return result

# # Example usage
# tensor_1 = torch.tensor([1, 3, 5, 6, 7, 4, 2, 5])
# tensor_2 = torch.tensor([6, 2])
# split_result = split_tensor(tensor_1, tensor_2)
# print(split_result)


# b = []
# for i in range(3):
#     a = []
#     b.append(a)
    
# print(len(b))

# k = torch.tensor([3, 3, 3, 3, 3, 3])
# print(k.size()[0])
# a = torch.tensor([[[1,1,1,1],
#                   [1,1,1,1]]])

# b = torch.tensor([[[3,3,3,3],
#                   [3,3,3,3],
#                   [3,3,3,3],
#                   [3,3,3,3],
#                   [3,3,3,3]]])


# print(a.size(), a)
# print(b.size(), b)

# b[0,3:5,:] = a[0]
# print(b.size(), b)
# myTensor = torch.rand((2,16,5))
# l = 5
# myTensor[0,l:,:] = 0
# print(myTensor)

# loss = torch.nn.BCEWithLogitsLoss()
# input = torch.randn(3, requires_grad=True)
# print(input.size(), input)
# target = torch.empty(3).random_(2)
# print(target.size(), target)
# output = loss(input, target)
# print(output)
# output.backward()
torch.manual_seed(1)
loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
input1 = torch.randn(1, 5, requires_grad=True)
input2 = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, -1 ,-1])

print(input1)
print(input2)
print(target)
output = loss(input1, input2, target)
print(output.item())
output.backward()

print(input1[0])



o1 = F.cosine_similarity(input1[0], input2[0], dim=0).item()
o1 = 1 - o1
print(o1)

o2 = F.cosine_similarity(input1[0], input2[1], dim=0).item()
o2 = max(0, o2)
print(o2)

o3 = F.cosine_similarity(input1[0], input2[2], dim=0).item()
o3 = max(0, o3)
print(o2)

print(np.mean([o1,o2,o3]))


print((1 - torch.triu(torch.ones((1, 5, 10)), diagonal=1)).bool())