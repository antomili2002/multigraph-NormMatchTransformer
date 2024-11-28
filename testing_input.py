

import torch
import torch.nn as nn

# Sample input
# t1 = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
# print(t1)
# t2 = ~torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=0)
# print(t2)
# t3 = t1 + t2
# print(t3)
# current_pos = 3
# for i in range(current_pos):
#     t3[i:,i] = False
# b = torch.arange(12).unsqueeze(0).expand(2, -1)
# print(b)
# br
# print(t3)
t1 = torch.tensor([[[2,3,4], [4,4, 4], [8,8,8]],[[1, 1, 1],[2,2,2], [0,0,0]]])
# print(t1.shape, t1)
# t1 = torch.rand([2, 10, 10])
# print(t1.shape, t1)
t2 = torch.tensor([[1, 0, 2],[1, 2, 0]])
print(t2.shape, t2)
for i in range(2):
    t1[i, :, :] = t1[i, t2[i], :]

print(t1)
# batch_size = t1.shape[0]

# # torch.max(t1[0], dim=-1)
# max_e, max_i = torch.max(t1[0], dim=-1)
# print(max_e, max_i)
# maxxx_e, maxxx_i = torch.max(max_e, dim=-1)
# print(maxxx_e, maxxx_i)
# print(torch.max(torch.max(t1[0], dim=-1)[0]))
# Attention matrix: [batch, seq_len, seq_len]
# nested_dict = {
#     1: {
#         'aeroplane': {9: [2, torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2], device='cuda:0', dtype=torch.int32)], 11: [2, torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1], device='cuda:0', dtype=torch.int32)], 5: [2, torch.tensor([2, 2, 2, 2, 2, 2,  2, 2, 1, 1], device='cuda:0', dtype=torch.int32)]},
#         'bicycle': {11: [2, torch.tensor([0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2], device='cuda:0', dtype=torch.int32)]},
#         'bird': {5: [2, torch.tensor([2, 0, 0, 2, 2], device='cuda:0', dtype=torch.int32)]},
#         # Rest of the data...
#     }
# }
# # for main_key, sub_dict in nested_dict.items():
# #     print(f"Main Key: {main_key}")
# #     for category, inner_dict in sub_dict.items():
# #         print(category)
# k = sorted(nested_dict[1]["aeroplane"].items())


# print(k)
# l = k[-1]
# print(l)

# a = torch.tensor([2,4,8,2])
# b = torch.tensor([0,1,0,2])

# print(a/4)


# t1 = torch.tensor([1, 1])
# t2 = torch.tensor([0, 1, 0, 0, 0])

# # Resize t1 to match the size of t2 and fill the rest with zeros
# t1_resized = torch.cat((t1, torch.zeros(0, dtype=t1.dtype)))
# print(t1_resized)
# def calculate_micro_f1_for_epoch(predicted_batches, actual_batches, num_classes):
#     # Initialize counters for the epoch
#     TP_epoch = torch.tensor(0, dtype=torch.int32)
#     FP_epoch = torch.tensor(0, dtype=torch.int32)
#     FN_epoch = torch.tensor(0, dtype=torch.int32)
    
#     # Iterate over batches
#     for predicted, actual in zip(predicted_batches, actual_batches):
#         for c in range(num_classes):
#             # Accumulate TP, FP, FN across all batches for all classes
#             TP_epoch += torch.sum((predicted == c) & (actual == c))
#             FP_epoch += torch.sum((predicted == c) & (actual != c))
#             FN_epoch += torch.sum((predicted != c) & (actual == c))
    
#     # Compute precision, recall, and F1 score (micro-averaged)
#     precision = TP_epoch / (TP_epoch + FP_epoch + 1e-8)  # Avoid division by zero
#     recall = TP_epoch / (TP_epoch + FN_epoch + 1e-8)
#     f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
#     return f1_score.item(), precision.item(), recall.item()

# # Example usage
# # Predicted and actual matchings (class representations) for batches
# predicted_batches = [torch.tensor([0, 2, 1, 2]), torch.tensor([1, 0, 2, 1])]
# actual_batches = [torch.tensor([0, 1, 1, 2]), torch.tensor([1, 0, 2, 2])]

# num_classes = 3  # Number of unique matchings/classes
# f1_score, precision, recall = calculate_micro_f1_for_epoch(predicted_batches, actual_batches, num_classes)

# print("Micro-Averaged F1 Score:", f1_score)
# print("Micro-Averaged Precision:", precision)
# print("Micro-Averaged Recall:", recall)
# import numpy as np
# a = np.array([1,1,0,1])
# b = np.array([1,0,1,0])

# print(a+b)
# predictions = torch.tensor([
#     [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ], device='cuda:0')

# ground_truth = torch.tensor([
#     [0, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [12, 9, 16, 4, 7, 18, 3, 0, 8, 6, 17, 13, 15, 5, 1, 14, 10, 2, 11]
# ], device='cuda:0')

# # Initialize the result dictionary
# result_dict = {}

# # Iterate through the batch
# for idx in range(predictions.size(0)): # Loop through batch size
#     pred_row = predictions[idx]
#     gt_row = ground_truth[idx]
#     length = pred_row.size(0)  # Length of the current row

#     # Generate a tensor where 1 indicates a wrong prediction, 0 indicates correct
#     comparison = (pred_row != gt_row).int()

#     # Add the result to the dictionary
#     if length not in result_dict:
#         result_dict[length] = []
#     result_dict[length].append(comparison.tolist())

# # Output the dictionary
# print(result_dict)
