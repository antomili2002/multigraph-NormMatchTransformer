import torch


t1 = torch.tensor([0, 0, 0, 0])
t2 = torch.tensor([1])

print(t1 + t2)

l = torch.triu(torch.ones(10, 10, dtype=torch.bool))

print(l)
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
