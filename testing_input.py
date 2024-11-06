import torch
import numpy as np


tensor_rows = []

def split_tensor(tensor_1, tensor_2):
    result = []
    start_index = 0

    for length in tensor_2:
        end_index = start_index + length
        result.append(tensor_1[start_index:end_index])
        start_index = end_index

    return result

# Example usage
tensor_1 = torch.tensor([1, 3, 5, 6, 7, 4, 2, 5])
tensor_2 = torch.tensor([6, 2])
split_result = split_tensor(tensor_1, tensor_2)
print(split_result)


# b = []
# for i in range(3):
#     a = []
#     b.append(a)
    
# print(len(b))

# k = torch.tensor([3, 3, 3, 3, 3, 3])
# print(k.size()[0])
a = torch.tensor([[[1,1,1,1],
                  [1,1,1,1]]])

b = torch.tensor([[[3,3,3,3],
                  [3,3,3,3],
                  [3,3,3,3],
                  [3,3,3,3],
                  [3,3,3,3]]])


print(a.size(), a)
print(b.size(), b)

b[0,3:5,:] = a[0]
print(b.size(), b)
# myTensor = torch.rand((2,16,5))
# l = 5
# myTensor[0,l:,:] = 0
# print(myTensor)