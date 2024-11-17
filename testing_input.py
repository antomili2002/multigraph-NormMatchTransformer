import torch

def calculate_micro_f1_for_epoch(predicted_batches, actual_batches, num_classes):
    # Initialize counters for the epoch
    TP_epoch = torch.tensor(0, dtype=torch.int32)
    FP_epoch = torch.tensor(0, dtype=torch.int32)
    FN_epoch = torch.tensor(0, dtype=torch.int32)
    
    # Iterate over batches
    for predicted, actual in zip(predicted_batches, actual_batches):
        for c in range(num_classes):
            # Accumulate TP, FP, FN across all batches for all classes
            TP_epoch += torch.sum((predicted == c) & (actual == c))
            FP_epoch += torch.sum((predicted == c) & (actual != c))
            FN_epoch += torch.sum((predicted != c) & (actual == c))
    
    # Compute precision, recall, and F1 score (micro-averaged)
    precision = TP_epoch / (TP_epoch + FP_epoch + 1e-8)  # Avoid division by zero
    recall = TP_epoch / (TP_epoch + FN_epoch + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1_score.item(), precision.item(), recall.item()

# Example usage
# Predicted and actual matchings (class representations) for batches
predicted_batches = [torch.tensor([0, 2, 1, 2]), torch.tensor([1, 0, 2, 1])]
actual_batches = [torch.tensor([0, 1, 1, 2]), torch.tensor([1, 0, 2, 2])]

num_classes = 3  # Number of unique matchings/classes
f1_score, precision, recall = calculate_micro_f1_for_epoch(predicted_batches, actual_batches, num_classes)

print("Micro-Averaged F1 Score:", f1_score)
print("Micro-Averaged Precision:", precision)
print("Micro-Averaged Recall:", recall)
