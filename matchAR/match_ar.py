import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from scipy.optimize import linear_sum_assignment

import utils.backbone
from matchAR.sconv_archs import SConv
from matchAR.positionalEmbedding import Pointwise2DPositionalEncoding
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from utils.evaluation_metric import make_perm_mat_pred
from utils.visualization import easy_visualize
from matchAR.nGPT_decoder import NGPT_DECODER
from matchAR.nGPT_encoder import NGPT_ENCODER


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def make_queries(h_s, h_t):
    n_s = h_s.size(dim=1)
    n_t = h_t.size(dim=1)

    queries = []

    for i in range(0,n_s):
        for j in range(0, n_t):
            query = torch.cat((h_s[:,i,:], h_t[:,j,:]), dim=-1)
            queries.append(query)
    queries = torch.stack(queries, dim=1)
    
    return queries

def pad_input(input_tensor, target_length=40):
    """
    Pads the input tensor to match the target length in the data point dimension.
    
    Parameters:
    - input_tensor: Input tensor of shape [batch_size, current_length, feature_dim].
    - target_length: The desired length for padding.

    Returns:
    - Padded tensor of shape [batch_size, target_length, feature_dim].
    """
    current_length = input_tensor.size(1)
    if current_length >= target_length:
        return input_tensor  # No padding needed if current length meets or exceeds target length

    # Padding to the right along the data point dimension (dim=1)
    padding_size = target_length - current_length
    padded_tensor = F.pad(input_tensor, (0, 0, 0, padding_size), "constant", 0)
    return padded_tensor

def create_source_masks(source_points, n_points, max_length=40):
    """
    Create masks for the source points tensor used in the TransformerDecoder.

    Parameters:
    - source_points: Tensor of shape [batch_size, seq_len, feature_dim].
    - n_points: List or tensor indicating the original length of each sequence before padding.
    - max_length: The maximum sequence length (after padding).

    Returns:
    - source_points_mask: Upper triangular mask of shape [seq_len, seq_len] (or None if not needed).
    - source_key_padding_mask: Padding mask of shape [batch_size, max_length].
    """
    batch_size, seq_len, _ = source_points.size()

    # Create target mask (if needed for causal attention)
    source_points_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(source_points.device)

    # Create key padding mask for source points
    source_key_padding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=source_points.device)
    for i, length in enumerate(n_points):
        source_key_padding_mask[i, length:] = True  # Mark padding positions with True

    return source_points_mask, source_key_padding_mask

def cosine_norm(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    Places vectors onto the unit-hypersphere

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    # calculate the magnitude of the vectors
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-6)
    # divide by the magnitude to place on the unit hypersphere
    return x / norm

class Scale(nn.Module):
    """
    A module that manages learnable scaling parameters to ensure different learning rates
    from the rest of the parameters in the model (see pages 5 and 19)
    
    Args:
        dim (int): Dimension of the scaling parameter
        scale (float): Initial scale value
        init (float): Initial value for the scaling parameter
        device (str, optional): Device to store the parameter on
    """
    def __init__(self, dim: int, heads: int = 1, scale: float = 1.0, init: float = 1.0, device=None):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
                      if device is None else device)
        self.init = init
        self.scale = scale
        self.s = nn.Parameter(torch.ones(heads, dim, device=self.device) * scale)
            # heads == 1 gives us a single regular vector
            # heads > 1 gets used in attention mechanism for different scaling vector for each head
    
    def forward(self):
        """Compute the effective scaling factor."""
        return self.s * (self.init / self.scale) # shape (heads, dim)

class ModelConfig:
    """
    Design your N-GPT here
    """
    dim: int = 128
    device: str = None
        # defaults to best available GPU/CPU
    num_layers: int = 6
    num_heads: int = 4 # number of heads in the multi-head attention mechanism
    mlp_hidden_mult: float = 4

class MatchARNet(utils.backbone.VGG16_bn):
    def __init__(self):
        super(MatchARNet, self).__init__()
        self.model_name = 'Transformer'
        self.psi = SConv(input_features=cfg.SPLINE_CNN.input_features, output_features=cfg.Matching_TF.d_model)
        # self.mlp = MLPQuery(cfg.Matching_TF.d_model, 1024, cfg.Matching_TF.d_model, batch_norm=cfg.Matching_TF.batch_norm)
        
        self.vgg_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(512, cfg.Matching_TF.d_model)

        self.s_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.t_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.cls_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        # self.scaled_mlp = MLP_scaled(cfg.Matching_TF.d_model*2, cfg.Matching_TF.d_model//2, cfg.Matching_TF.d_model)      
        
        self.pos_encoding = Pointwise2DPositionalEncoding(cfg.Matching_TF.d_model, 256, 256).cuda()

        # self.tf_encoder_layer = nn.TransformerEncoderLayer(d_model= cfg.Matching_TF.d_model, 
        #                                                    nhead= cfg.Matching_TF.n_head, 
        #                                                    batch_first=True)
        # self.tf_decoder_layer = nn.TransformerDecoderLayer(d_model= cfg.Matching_TF.d_model,
        #                                                    nhead=cfg.Matching_TF.n_head,
        #                                                    batch_first=True)
        # self.tf_encoder = nn.TransformerEncoder(self.tf_encoder_layer, num_layers=cfg.Matching_TF.n_encoder)
        # self.tf_decoder = nn.TransformerDecoder(self.tf_decoder_layer, num_layers=cfg.Matching_TF.n_decoder)
        
        nGPT_decoder_config = ModelConfig()
        nGPT_decoder_config.dim = cfg.Matching_TF.d_model
        nGPT_decoder_config.num_layers = cfg.Matching_TF.n_decoder
        nGPT_decoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        nGPT_decoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        self.n_gpt_decoder = NGPT_DECODER(nGPT_decoder_config)
        
        self.n_gpt_decoder_2 = NGPT_DECODER(nGPT_decoder_config)
        
        # nGPT_encoder_config = ModelConfig()
        # nGPT_encoder_config.dim = cfg.Matching_TF.d_model
        # nGPT_encoder_config.num_layers = cfg.Matching_TF.n_encoder
        # nGPT_encoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        # nGPT_encoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        # self.n_gpt_encoder = NGPT_ENCODER(nGPT_encoder_config)
        
        # self.prot_MLP = MLP_prototype(cfg.Matching_TF.d_model)
        
        self.mlp_out = MLP([cfg.Matching_TF.d_model, 384], cfg.Matching_TF.d_model, l2_scaling=False)
        # self.mlp_out_2 = MLP([cfg.Matching_TF.d_model, 512, 1024], 512, l2_scaling=True)
        # self.mlp_out = MLP([cfg.Matching_TF.d_model, 512, 1024], 512, batch_norm=False)
        # self.mlp_out_2 = MLP([cfg.Matching_TF.d_model, 512, 1024], 512, batch_norm=False)
        self.w_cosine = PairwiseWeightedCosineSimilarity(cfg.Matching_TF.d_model)
        
        self.global_state_dim = 1024

        # matched encoding
        # self.matched_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        # mask_match encoding
        # self.mask_match_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
    
    def normalize_linear(self, module):
        """
        Helper method to normalize Linear layer weights where one dimension matches model dim
        """
        # Find the dimension that matches cfg.dim
        dim_to_normalize = None
        for dim, size in enumerate(module.weight.shape):
            if size == cfg.Matching_TF.d_model:
                dim_to_normalize = dim
                break
        
        if dim_to_normalize is not None:
            # Normalize the weights
            module.weight.data = cosine_norm(module.weight.data, dim=dim_to_normalize)
    
    def enforce_constraints(self):
        """
        Enforces constraints after each optimization step:
        2. Cosine normalization on Linear layer weights where one dimension matches model dim
        """
        # for layer in self.n_gpt_encoder.layers:
        #     layer.alpha_A.s.data.abs_()
        #     layer.alpha_M.s.data.abs_()
            
        for layer in self.n_gpt_decoder.layers:
            layer.alpha_A.s.data.abs_()
            layer.alpha_C.s.data.abs_()
            layer.alpha_G.s.data.abs_()
            layer.alpha_M.s.data.abs_()
        
        for layer in self.n_gpt_decoder_2.layers:
            layer.alpha_A.s.data.abs_()
            layer.alpha_C.s.data.abs_()
            layer.alpha_G.s.data.abs_()
            layer.alpha_M.s.data.abs_()
        # Cosine normalize relevant Linear layers
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                self.normalize_linear(module)
    
    # def update_queries(self, Q, in_training, eval_pred_points, n_points, all_targets):
    #     if in_training is True:
    #         return Q
    #     B, _, _ = Q.size()
    #     new_queries = Q
    #     for i in range(B):
    #         new_queries[i, n_points[i]:, :] = 0
        
    #     if len(eval_pred_points[0]) > 0:
    #         for i in range(B):
    #             current_n_points = n_points[i]
    #             predicted_targets = eval_pred_points[i]
    #             selected_targets = all_targets[i,predicted_targets, :]
    #             print("----------------")
    #             print(predicted_targets)
    #             print(selected_targets.size(), selected_targets)
    #             new_queries[i, current_n_points:(current_n_points+len(predicted_targets)), :] = selected_targets
                
                
    #     return new_queries
    
    
    def update_order(self, source_nodes, input_order):
        B, _, _ = source_nodes.shape
        for b in range(B):
            source_nodes[b, :, :] = source_nodes[b, input_order[b], :]
        return source_nodes
    

    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        n_points_sample, 
        perm_mats,
        eval_pred_points=None,
        in_training=True,
        input_order=None,
        matched_points_mask=None,
        matched_padding_mask_hs=None,
        matched_padding_mask_ht=None,
    ):
        batch_size = graphs[0].num_graphs
        global_list = []
        orig_graph_list = []
        node_feat_list = []
        # for visualisation purposes only
        graph_list = []
        global_feat = 0
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            # with torch.no_grad():
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)
            
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)

            node_features = torch.cat((U, F), dim=-1)
            graph.x = node_features
            # for visualisation purposes only
            graph_list.append(graph.to_data_list())

            # node + edge features from vgg
            vgg_features = self.vgg_to_node_dim(node_features)
            
            # splineCNN spatial features 
            h = self.psi(graph)

            h_res = h + vgg_features
                            
            (h_res, mask) = to_dense_batch(h_res, graph.batch, fill_value=0)

            if cfg.Matching_TF.pos_encoding:
                h_res = h_res + self.pos_encoding(p)
                
            global_feature = self.final_layers(edges)[0].reshape((nodes.shape[0], -1))
            global_feature = self.glob_to_node_dim(global_feature)
            global_feature = global_feature + self.cls_enc
            global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            
            # global_feature = self.linear_cls(global_feature)
            
            h_res = torch.cat([global_feature, h_res], dim=1)

            global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
            mask = torch.cat([global_feature_mask, mask], dim=1)
            
            # global_feature = cosine_norm(global_feature).squeeze(1).unsqueeze(0)
            # if cfg.Matching_TF.global_feat:
            #     # with torch.no_grad():
            #     global_feature = self.final_layers(edges)[0].reshape((nodes.shape[0], -1))
            #     global_feature = self.glob_to_node_dim(global_feature)
            #     global_feature = global_feature + self.cls_enc
            #     global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            #     h_res = torch.cat([global_feature, h_res], dim=1)

            #     global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
            #     mask = torch.cat([global_feature_mask, mask], dim=1)


            orig_graph_list.append((h_res,mask))

        h_s, s_mask = orig_graph_list[0]
        h_t, t_mask = orig_graph_list[1]

        assert h_s.size(0) == h_t.size(0), 'batch-sizes are not equal'
        
        
        
        
        # if cfg.Matching_TF.global_feat != True:
        #     (B, N_s, D), N_t = h_s.size(), h_t.size(1)
        #     query_mask = ~(s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t))
        #     queries = make_queries(h_s, h_t)

        # else:
        #     (B, N_s, D), N_t = h_s.size(), h_t.size(1)
        #     N_s -= 1
        #     N_t -= 1 
        #     query_mask = ~(s_mask[:,:-1].view(B, N_s, 1) & t_mask[:,:-1].view(B, 1, N_t))
        #     queries = make_queries(h_s[:,:-1,:], h_t[:,:-1,:])
        
        # query_mask = query_mask.view(B, -1)
        # print(query_mask.size(), query_mask)
        # print(n_points_sample)
        batch_size, seq_len, _ = h_s.shape
        padding_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool).to(h_s.device)
        if in_training is True:
            for idx, e in enumerate(n_points_sample):
                if cfg.Matching_TF.global_feat:
                    s_mask[idx, e+1:] = False
                    h_s[idx, e+1:, :] = 0
                    
                    t_mask[idx, e+1:] = False
                    h_t[idx, e+1:, :] = 0
                    
                    padding_mask[idx, :, e+1:] = 1
                    padding_mask[idx, e+1:, :] = 1
                else:
                    s_mask[idx, e:] = False
                    h_s[idx, e:, :] = 0
                    
                    t_mask[idx, e:] = False
                    h_t[idx, e:, :] = 0
          
        source_points_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(h_s.device)#(1 - torch.triu(torch.ones((batch_size, sample_size_each, sample_size_each)), diagonal=1)).bool()
        if cfg.Matching_TF.global_feat:
            source_points_mask[0, :] = False
            
        if in_training == False:
            if cfg.Matching_TF.global_feat:
                h_s[:,1:, :] = self.update_order(h_s[:,1:, :], input_order)
                diagonal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0).to(h_s.device)
                res_mask = source_points_mask + diagonal_mask
                for j in range(eval_pred_points):
                    res_mask[j:,j] = False
                res_mask[:,0] = False
                source_points_mask = res_mask
            else:
                h_s = self.update_order(h_s, input_order)
                diagonal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0).to(h_s.device)
                res_mask = source_points_mask + diagonal_mask
                for j in range(eval_pred_points):
                    res_mask[j:,j] = False
                source_points_mask = res_mask
        
        # print(n_points)
        # print(s_mask.shape, s_mask)
        # print(h_s.shape, h_s)
        # br
        # input = torch.cat((h_s + self.s_enc, h_t + self.t_enc), dim=1)
        # encoder_output = self.tf_encoder(src=input, src_key_padding_mask=S_mask)
        
        
        #Encoder-Decoder
        # S_mask = ~torch.cat((s_mask, t_mask), dim=1)
        # input = torch.cat((h_s, h_t), dim=1)
        # _, encoder_memory = self.n_gpt_encoder(input, S_mask)
        
        # sample_size_each = encoder_memory.size()[1] // 2 #Get the amount of concatenated points
        # # #split context sensitiv output from encoder into source and target patches
        # source_points = encoder_memory[:, :sample_size_each, :].to(encoder_memory.device)
        # target_points = encoder_memory[:, sample_size_each:, :].to(encoder_memory.device)
        
        
        
        
        if in_training == False:
            #Encoder-Decoder
            # hs_dec_output = self.n_gpt_decoder(source_points, source_points_mask, matched_padding_mask_ht, target_points)
            
            #Decoder-Decoder
            hs_dec_output = self.n_gpt_decoder(h_s, source_points_mask, matched_padding_mask_ht, h_t)
            ht_dec_output = self.n_gpt_decoder_2(h_t, matched_points_mask, matched_padding_mask_hs, h_s, is_eval=True)
        else:
            #Encoder-Decoder
            # hs_dec_output = self.n_gpt_decoder(source_points, source_points_mask, padding_mask, target_points)
            
            #Decoder-Decoder
            hs_dec_output = self.n_gpt_decoder(h_s, source_points_mask, padding_mask, h_t)
            ht_decoder_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool).to(h_t.device)
            ht_dec_output = self.n_gpt_decoder_2(h_t, ht_decoder_mask, padding_mask, h_s)
            
        # paired_global_feat = torch.cat([hs_dec_output[ :, 0, :], ht_dec_output[ :, 0, :]], dim=-2).to(hs_dec_output.device)
        # paired_global_feat = self.scaled_mlp(paired_global_feat)
        
        #Encoder-Decoder
        # hs_dec_output = hs_dec_output[:, 1:, :]
        # target_points = cosine_norm(target_points)
        # ht_dec_output = target_points[:, 1:, :]
        
        #Decoder-Decoder
        hs_dec_output = hs_dec_output[:, 1:, :]
        ht_dec_output = ht_dec_output[:, 1:, :]
        
        prototype_score = []
        if in_training == True:
            # print(hs_dec_output)
            # print(perm_mats[0].shape)
            match_idx = torch.argmax(perm_mats[0], dim=-1)
            # print(match_idx.shape)
            for b in range(hs_dec_output.shape[0]): #Batch size
                current_hs = hs_dec_output[b, :n_points[0][b].item(), :]
                current_ht = ht_dec_output[b, :n_points[0][b].item(), :]
                current_idx = match_idx[b, :n_points[0][b].item()]
                reOrdered_ht = current_ht[current_idx]
                matches_expanded = (current_hs * reOrdered_ht).unsqueeze(0)#torch.concat([current_hs, reOrdered_ht], dim=-1).unsqueeze(0)
                matches_expanded = cosine_norm(matches_expanded)
                matches_sim = torch.bmm(matches_expanded, matches_expanded.transpose(-2, -1))
                prototype_score.append(matches_sim)
        
        
        # h_t_norm = cosine_norm(h_t)
        sim_score = self.w_cosine(hs_dec_output, ht_dec_output) #self.w_cosine(hs_dec_output, h_t_norm)
        
        hs_dec_output_ = hs_dec_output.reshape(hs_dec_output.shape[0] * hs_dec_output.shape[1], hs_dec_output.shape[2])
        ht_dec_output_ = ht_dec_output.reshape(ht_dec_output.shape[0] * ht_dec_output.shape[1], ht_dec_output.shape[2])
        
        sim_score_all = self.w_cosine(hs_dec_output_, ht_dec_output_)
        
        # prototype_score = torch.bmm(ht_dec_output, ht_dec_output.transpose(1, 2)) #torch.bmm(h_t_norm, h_t_norm.transpose(1, 2))
        
        # Seoncd Loss for global(class) sim-score
        # paired_global_feat = cosine_norm(paired_global_feat)
        
        # split_size = paired_global_feat.shape[0] // 2
        # gloabl_feat_score = paired_global_feat[:split_size, :] @ paired_global_feat[split_size:, :].transpose(-1, -2)
        
        
        return sim_score, sim_score_all, prototype_score, hs_dec_output, ht_dec_output #gloabl_feat_score  #target_points, hs_dec_output
        


class MLP_prototype(nn.Module):
    def __init__(self, model_dim):
        super(MLP_prototype, self).__init__()
        self.layer1 = nn.Linear(model_dim * 2, model_dim)

    def forward(self, x, dropout=None):
        # Feedforward
        x = self.layer1(x)
        if dropout is not None:
            x = torch.nn.functional.dropout(x, p=dropout)
        
        # if self.l2_scaling:
        #     # Apply L2 normalization to the final layer output
        #     output = F.normalize(output, p=2, dim=-1)
        return x


class MLP(nn.Module):
    def __init__(self, h_sizes, out_size, l2_scaling):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        self.l2_scaling = l2_scaling
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = layer(x)
            x = F.relu(x)
        output = self.out(x)
        # if self.l2_scaling:
        #     # Apply L2 normalization to the final layer output
        #     output = F.normalize(output, p=2, dim=-1)
        return output 

class MLP(nn.Module):
    def __init__(self, h_sizes, out_size, l2_scaling):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        self.l2_scaling = l2_scaling
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = layer(x)
            x = F.relu(x)
        output = self.out(x)
        # if self.l2_scaling:
        #     # Apply L2 normalization to the final layer output
        #     output = F.normalize(output, p=2, dim=-1)
        return output
# class MLP(nn.Module):
#     def __init__(self, h_sizes, out_size, batch_norm):
#         super(MLP, self).__init__()
#         self.hidden = nn.ModuleList()
#         self.bn_layers = nn.ModuleList()
#         self.batch_norm = batch_norm
#         for k in range(len(h_sizes)-1):
#             self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
#             if batch_norm:
#                 self.bn_layers.append(nn.BatchNorm1d(h_sizes[k+1]))
#         self.out = nn.Linear(h_sizes[-1], out_size)
    
#     def forward(self, x):

#         # Feedforward
#         for i in range(len(self.hidden)):
#             if self.batch_norm:
#                 x = torch.transpose(self.bn_layers[i](torch.transpose(self.hidden[i](x),1,2)),1,2)
#             else:
#                     x = self.hidden[i](x)
#             x = nn.functional.relu(x)
#         output = self.out(x)
#         return output

class MLPQuery(nn.Module):
    def __init__(self, node_dim, hidden_size, hidden_out, batch_norm):
        super(MLPQuery, self).__init__()
        self.lin1 = nn.Linear(2 * node_dim, hidden_size)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_out)

    def forward(self, x):
            x = self.lin1(x)
            if self.batch_norm:
                x = self.bn(torch.transpose(x, 1, 2))
                x = torch.transpose(x, 1, 2)
            x = nn.functional.relu(x)
            out = self.lin2(x)
            return out



class PairwiseWeightedCosineSimilarity(nn.Module):
    def __init__(self, node_feature_dim):
        super(PairwiseWeightedCosineSimilarity, self).__init__()
        # Initialize weights with ones for each feature dimension
        self.w = nn.Parameter(torch.ones(1, 1, node_feature_dim))
    
    def forward(self, x, y):
        # x and y have shape [batch_size, nodes, node_feature]
        
        # Apply weights
        x_weighted = x * self.w  # Shape: [batch_size, nodes_x, node_feature]
        y_weighted = y * self.w  # Shape: [batch_size, nodes_y, node_feature]
        
        y_weighted_transposed = y_weighted.transpose(-2, -1)  # Shape: [batch_size, node_feature, nodes_y]
        numerator = torch.bmm(x_weighted, y_weighted_transposed)  # Shape: [batch_size, nodes_x, nodes_y]
        
        x_norm = torch.norm(x_weighted, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_x]
        y_norm = torch.norm(y_weighted, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_y]
        #epsilon = 1e-8  # To prevent division by zero
        #x_norm = x_norm + epsilon
        #y_norm = y_norm + epsilon
        
        denominator = torch.bmm(x_norm.unsqueeze(2), y_norm.unsqueeze(1))  # Shape: [batch_size, nodes_x, nodes_y]
        
        # Compute cosine similarity matrix
        cosine_similarity = numerator / denominator  # Shape: [batch_size, nodes_x, nodes_y]
        #cosine_similarity = torch.clamp(cosine_similarity, -1 + epsilon, 1 - epsilon)
        
        return cosine_similarity
        
class MLP_scaled(nn.Module):
    """
    Multilayer Perceptron (MLP) module with optional gating and dropout.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output features.
        device (str or torch.device): Device to run the module on.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device = None
    ):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)

        # the up, down, and gate projections
        self.Wup = nn.Linear(input_dim, hidden_dim, bias=False, device=self.device)
        self.Wgate = nn.Linear(input_dim, hidden_dim, bias=False, device=self.device)
        self.Wdown = nn.Linear(hidden_dim, output_dim, bias=False, device=self.device)

        # this flag designates Wdown to have a different parameter initialization as defined in model.py
        self.Wdown.GPT_scale_init = 1 

        # the learnable scaling factors
        self.s_u = Scale(hidden_dim, device=device)
        self.s_v = Scale(hidden_dim, device=device)

        # the varaince-controlling scaling term, needed to benefit from SiLU (see appendix A.1)
        self.scale = math.sqrt(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        # our up & gate projections
        u = self.Wup(x) # (batch_size, seq_len, hidden_dim)
        v = self.Wgate(x)
        # scale them
        u = u * self.s_u()
        v = v * self.s_v() * self.scale 
        # now perform the nonlinearity gate
        hidden = u * F.silu(v) # (batch_size, seq_len, hidden_dim)
        return self.Wdown(hidden) # (batch_size, seq_len, output_dim)
