import torch

class GS_InfoNCE_Loss(torch.nn.Module):
    def __init__(self, temperature, noise_scale, lambda_balance):
        
        super(GS_InfoNCE_Loss, self).__init__()
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.lambda_balance = lambda_balance

    def forward(self, source_Points, target_Points, pos_indices, n_points):
        
        device = source_Points.device
        batch_size, num_points, feature_size = source_Points.shape
        
        sim_numerator = torch.bmm(source_Points, target_Points.transpose(1, 2))
        
        source_normed = torch.norm(source_Points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(2)
        target_normed = torch.norm(target_Points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(1)
        sim_denominator = torch.bmm(source_normed, target_normed)
        
        cosine_sim = sim_numerator / sim_denominator
        
        sim_tensor = []
        for i in range(batch_size):
            sim_tensor.append(cosine_sim[i, :n_points[i], :])
        
        sim_tensor = torch.concat(sim_tensor, dim=0).to(device)
        
        pos_idx_mask = torch.zeros((sim_tensor.shape[0], sim_tensor.shape[1]), dtype=torch.bool).to(sim_tensor.device)
        rows = torch.arange(sim_tensor.shape[0]).to(sim_tensor.device)
        pos_idx_mask[rows, pos_indices] = 1
        
        pos_score = torch.masked_select(sim_tensor, pos_idx_mask).reshape(-1, 1)
        pos_score = pos_score / self.temperature
        pos_score = torch.exp(pos_score)
        
        trans_cosine_sim = cosine_sim.transpose(1, 2)
        trans_sim_tensor = []
        for i in range(batch_size):
            trans_sim_tensor.append(trans_cosine_sim[i, :n_points[i], :])
        trans_sim_tensor = torch.concat(trans_sim_tensor, dim=0).to(device)
        
        trans_sim_tensor = trans_sim_tensor / self.temperature
        trans_sim_tensor = torch.exp(trans_sim_tensor)
        trans_sim_tensor_sum = torch.sum(trans_sim_tensor, dim=-1).reshape(-1, 1)
        
        
        noise_vectors = torch.normal(
            mean=0.0,
            std=self.noise_scale,
            size=(num_points, feature_size)
        ).to(device)
        
        filtered_source_points = []
        for i in range(batch_size):
            filtered_source_points.append(source_Points[i, :n_points[i], :])
        filtered_source_points = torch.concat(filtered_source_points, dim=0).to(device)
            
        filtered_source_points_normed = torch.norm(filtered_source_points, p=2, dim=-1).clamp(min=1e-8).unsqueeze(1)
        noise_vectors_normed = torch.norm(noise_vectors, p=2, dim=-1).clamp(min=1e-8).unsqueeze(0)
        
        noise_numerator = torch.matmul(filtered_source_points, noise_vectors.transpose(0, 1))
        noise_denominator = torch.matmul(filtered_source_points_normed, noise_vectors_normed)
        
        noise_term = noise_numerator / noise_denominator
        noise_term = noise_term / self.temperature
        noise_term = torch.exp(noise_term)
        noise_term_sum = torch.sum(noise_term, dim=-1).reshape(-1, 1)
        noise_term_sum = self.lambda_balance * noise_term_sum
        
        loss = -torch.log(pos_score / (trans_sim_tensor_sum + noise_term_sum))
        loss = torch.mean(loss)
        return loss
        
        
if __name__ == '__main__':
    source_Points = torch.rand(2, 10, 512)
    target_Points = torch.rand(2, 10, 512)
    pos_indices = torch.tensor([0, 1, 3, 2, 6, 7, 8, 9, 0, 1, 3, 2, 6, 7, 8, 9, 5])
    n_points = torch.tensor([8, 9])
    loss = GS_InfoNCE_Loss(0.1, 1, 1)
    print(loss(source_Points, target_Points, pos_indices, n_points))