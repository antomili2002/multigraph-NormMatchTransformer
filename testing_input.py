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

batch_size = 4
num_points1 = 2  # number of points in input1 per batch
num_points2 = 3  # number of points in input2 per batch
feature_dim = 5

input1 = torch.randn(batch_size, num_points1, feature_dim, requires_grad=True)
input2 = torch.randn(batch_size, num_points2, feature_dim, requires_grad=True)
target = torch.tensor([[[1, -1, -1], [-1, 1, -1]]] * batch_size)

# Initialize total loss
total_loss = 0


for b in range(batch_size):
    batch_loss = 0
    for i in range(num_points1):
        # Compute cosine similarity of input1[b, i] with all points in input2[b]
        cosine_similarities = F.cosine_similarity(input1[b, i].unsqueeze(0), input2[b])
        
        # Apply target for this specific input1[b, i] row
        losses = torch.where(target[b, i] == 1, 1 - cosine_similarities, torch.clamp(cosine_similarities, min=0))
        
        # Accumulate the mean loss for this input1[b, i] with all points in input2[b]
        batch_loss += losses.mean()
        
    # Average loss across all points in input1 for the batch and accumulate
    total_loss += batch_loss / num_points1

# Average loss across the entire batch
final_loss = total_loss / batch_size
final_loss.backward()
print("Custom Cosine Embedding Loss with batch dimension:", final_loss.item())

# Backward pass

# Backward pass



# o1 = F.cosine_similarity(input1[0], input2[0], dim=0).item()
# o1 = 1 - o1
# # print(o1)

# o2 = F.cosine_similarity(input1[0], input2[1], dim=0).item()
# o2 = max(0, o2)
# # print(o2)

# o3 = F.cosine_similarity(input1[0], input2[2], dim=0).item()
# o3 = max(0, o3)
# # print(o2)

# o1_2 = F.cosine_similarity(input1[1], input2[0], dim=0).item()
# o1_2 = max(0, o1_2)
# # print(o1_2)

# o2_2 = F.cosine_similarity(input1[1], input2[1], dim=0).item()
# o2_2 = 1 - o2_2
# # print(o2_2)

# o3_2 = F.cosine_similarity(input1[1], input2[2], dim=0).item()
# o3_2 = max(0, o3_2)
# # print(o3_2)


# p1 = np.mean([o1,o2,o3])
# p2 = np.mean([o1_2,o2_2,o3_2])

# print(p1)
# print(p2)
# print((p1+p2)/2)

print()

# print((1 - torch.triu(torch.ones((1, 5, 10)), diagonal=1)).bool())