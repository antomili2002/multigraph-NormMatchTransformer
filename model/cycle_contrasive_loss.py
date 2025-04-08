import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class CycleContrastiveLoss(nn.Module):
    """
    Implements a global InfoNCE-based contrastive loss with cycle consistency
    and a prototype loss for multigraph keypoint matching.
    
    - For every anchor keypoint from graph i, all other keypoints from other graphs
      are used as negatives.
    - Positive scores are extracted from ground truth permutation matrices.
    - Cycle consistency is enforced through triplet permutations.
    - Prototype loss promotes diversity within graph keypoint embeddings.
    """
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temperature)), requires_grad=True)

    def forward(self, similarity_matrices, perm_mats):
        """
        Args:
            similarity_matrices: List[List[Tensor]], where sim[i][j] is [B, K, K] cosine sim matrix from graph i to j
            perm_mats: List[List], where perm_mats[i][j] is [B, K, K] permutation matrix from i -> j
        
        Returns:
            Scalar contrastive + cycle + prototype loss
        """
        n = len(similarity_matrices)
        assert isinstance(similarity_matrices[0][0], torch.Tensor), f"sim matrix is not of instance torch.Tensor, but {type(similarity_matrices[0][0])}"
         
        B, K, _ = similarity_matrices[0][0].shape
        temp = self.log_temp.exp()

        total_loss = 0
        pairwise_count = 0

        # --- Global InfoNCE: Use all graphs as anchors
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sim = similarity_matrices[i][j] / temp        # [B, K, K]
                pos_mask = perm_mats[i][j].to(sim.device)     # [B, K, K]
                
                if pos_mask.ndim == 2:
                    pos_mask = pos_mask.unsqueeze(0)
                if pos_mask.shape[0] == 1 and sim.shape[0] > 1:
                    pos_mask = pos_mask.expand(sim.shape[0], -1, -1)
                
                # handle shape mismatch (CLS token etc.)
                if sim.shape[1] != pos_mask.shape[1] or sim.shape[2] != pos_mask.shape[2]:
                    min_k1 = min(sim.shape[1], pos_mask.shape[1])
                    min_k2 = min(sim.shape[2], pos_mask.shape[2])
                    sim = sim[:, :min_k1, :min_k2]
                    pos_mask = pos_mask[:, :min_k1, :min_k2]
                
                assert sim.shape == pos_mask.shape, f"Mismatch: sim={sim.shape}, pos_mask={pos_mask.shape}"
                
                #print("Loss")
                #print("---------------------------------------------")
                #print(f"sim_matrix: {sim.shape}")
                #print(f"perm_matrix: {perm_mats[i][j].shape}")
                #print(f"pos_mask (cropped): {pos_mask.shape}")
                #if i == 0 and j == 1:
                #    print(f"sim[{i}][{j}].shape: {sim.shape}, perm_mats[{i}][{j}].shape: {perm_mats[i][j].shape}")

                
                pos_scores = (sim * pos_mask).sum(dim=-1)     # [B, K]
                denom = torch.logsumexp(sim, dim=-1)          # [B, K]
                total_loss += -torch.mean(pos_scores - denom) # scalar
                pairwise_count += 1


        # Try out without cycle consistency term
        """
        # --- Cycle Consistency: P_ik ≈ P_ij * P_jk
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len({i, j, k}) < 3:
                        continue
                    P_ij = perm_mats[i][j]  # [B, K, K]
                    P_jk = perm_mats[j][k]  # [B, K, K]
                    P_ik = perm_mats[i][k]  # [B, K, K]
                    P_pred = torch.bmm(P_ij, P_jk)  # [B, K, K]
                    total_loss += F.mse_loss(P_pred, P_ik)
                    pairwise_count += 1
        """

        # --- Prototype Loss: within-graph dissimilarity
        proto_loss = 0.0
        for i in range(n):
            sim_matrix = similarity_matrices[i][i]
            I = torch.eye(sim_matrix.shape[1], device=sim_matrix.device).unsqueeze(0)  # [1, K, K]
            
            #print("Prototype Loss")
            #print(f"sim matrix[{i}][{i}]: {sim_matrix.shape}")
            #print(f"Identity: {I.shape}")
            
            sim_matrix = sim_matrix - 2 * I
            max_sim, _ = torch.max(sim_matrix, dim=-1)  # [B, K]
            proto_loss += max_sim.mean()
        proto_loss = proto_loss / n

        return (total_loss / pairwise_count) + proto_loss

