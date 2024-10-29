import torch

# Beispiel-Tensoren in torch
tensor_1 = torch.zeros(4, 4, 4)
tensor_2 = torch.zeros(4, 4, 4)

# Setze die Einsen im ersten Tensor
tensor_1[0, 0, 2] = 1
tensor_1[0, 1, 3] = 1
tensor_1[0, 2, 1] = 1
tensor_1[0, 3, 0] = 1

tensor_1[1, 0, 1] = 1
tensor_1[1, 1, 2] = 1
tensor_1[1, 2, 1] = 1
tensor_1[1, 3, 3] = 1

tensor_1[2, 0, 1] = 1
tensor_1[2, 1, 1] = 1
tensor_1[2, 2, 1] = 1
tensor_1[2, 3, 3] = 1

tensor_1[3, 0, 2] = 1
tensor_1[3, 1, 1] = 1
tensor_1[3, 2, 1] = 1
tensor_1[3, 3, 2] = 1

# Setze die Einsen im zweiten Tensor, inklusive einer Abweichung
tensor_2[0, 0, 3] = 1
tensor_2[0, 1, 3] = 1
tensor_2[0, 2, 1] = 1  
tensor_2[0, 3, 0] = 1

tensor_2[1, 0, 1] = 1
tensor_2[1, 1, 2] = 1
tensor_2[1, 2, 1] = 1
tensor_2[1, 3, 3] = 1

tensor_2[2, 0, 1] = 1
tensor_2[2, 1, 1] = 1
tensor_2[2, 2, 1] = 1
tensor_2[2, 3, 3] = 1

tensor_2[3, 0, 2] = 1
tensor_2[3, 1, 2] = 1
tensor_2[3, 2, 0] = 1  
tensor_2[3, 3, 0] = 1



"""
tensor_1:
[
 [0, 0, 1, 0],
 [0, 0, 0, 1],
 [0, 1, 0, 0],
 [1, 0, 0, 0]
]  
tensor_2:
[
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 1, 0, 0],
 [1, 0, 0, 0]
]    

"""


# Ein Datenpunkt ohne Fehler
# tensor_2[1, 0, 2] = 1
# tensor_2[1, 1, 3] = 1
# tensor_2[1, 2, 1] = 1
# tensor_2[1, 3, 0] = 1  # Keine Abweichungen

# Finde die Positionen der 1 in tensor_1 und tensor_2 entlang der letzten Dimension
pos_1 = torch.argmax(tensor_1, dim=2)  # Position der 1 in tensor_1
pos_2 = torch.argmax(tensor_2, dim=2)  # Position der 1 in tensor_2

# Finde die Abweichungen (wo die Positionen unterschiedlich sind) innerhalb des gleichen Batch-Datenpunkts
diff_mask = (pos_1 != pos_2)  # Vergleiche den gleichen Batch

print(diff_mask)
# Erstelle eine Maske, die anzeigt, in welchen Datenpunkten (Batch-Dimension) ein Fehler vorliegt
batch_error_mask = torch.any(diff_mask, dim=1)
print(batch_error_mask)

true_indices = torch.nonzero(batch_error_mask, as_tuple=True)[0]
print(true_indices)


# print(torch.min(torch.nonzero(diff_mask[3], as_tuple=True)[0]))


my_list = []
for i in true_indices:
    my_tensor = tensor_1[i].clone()
    print(my_tensor)
    min_pos_err = torch.min(torch.nonzero(diff_mask[i], as_tuple=True)[0])
    print(min_pos_err)
    my_tensor[min_pos_err] = tensor_2[i][min_pos_err].clone()
    print(my_tensor)
    
    

"""
tensor([[ True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [ True, False, False, False]]) row: welcher batch, col: welcher Datenpunkt
tensor([ True, False, False,  True]) #
tensor([0, 3]) # Welche Bathes sind betroffen
"""


# Hinzufügen wird dann über self.length geregelt, sodass der Wrapper-Datensatz vergrößert wird indem die größe erhöht wird und sobal ein index über der eigentlichen größe hinausgeht, dass es aus den hinzugefügten daten dann samplet

