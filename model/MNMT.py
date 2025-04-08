import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt

import utils.backbone
from model.sconv_archs import SConv
from model.positionalEmbedding import Pointwise2DPositionalEncoding
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from utils.evaluation_metric import make_perm_mat_pred
from utils.visualization import easy_visualize
from model.nGPT_decoder import NGPT_DECODER
from model.nGPT_encoder import NGPT_ENCODER


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

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

class MNMT(utils.backbone.VGG16_bn):
    def __init__(self, num_graphs):
        super(MNMT, self).__init__()
        self.model_name = 'Transformer'
        self.num_graphs = num_graphs
        self.psi = SConv(input_features=cfg.SPLINE_CNN.input_features, output_features=cfg.Matching_TF.d_model)
        # self.mlp = MLPQuery(cfg.Matching_TF.d_model, 1024, cfg.Matching_TF.d_model, batch_norm=cfg.Matching_TF.batch_norm)
        
        self.vgg_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(512, cfg.Matching_TF.d_model)
        
        self.pos_encoding = Pointwise2DPositionalEncoding(cfg.Matching_TF.d_model, 256, 256).cuda()
        
        nGPT_decoder_config = ModelConfig()
        nGPT_decoder_config.dim = cfg.Matching_TF.d_model
        nGPT_decoder_config.num_layers = cfg.Matching_TF.n_decoder
        nGPT_decoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        nGPT_decoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        self.n_gpt_decoder = NGPT_DECODER(nGPT_decoder_config)
        
        # TO-DO: weight sharing between decoders
        self.graph_decoders = nn.ModuleList([NGPT_DECODER(nGPT_decoder_config) for _ in range(num_graphs)])
        
        self.w_cosine = PairwiseWeightedCosineSimilarity(cfg.Matching_TF.d_model)
        
        self.similarity_matrices = [] # pairewise cosine similarity storage
    
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
        for decoder in self.graph_decoders:
            decoder.enforce_constraints()
    
    
    
    def update_order(self, source_nodes, input_order):
        B, _, _ = source_nodes.shape
        for b in range(B):
            source_nodes[b, :, :] = source_nodes[b, input_order[b], :]
        return source_nodes
    

    def forward(self, images, points, graphs, n_points, n_points_sample, perm_mats, in_training=True):
        """
        Forward pass for multiple graphs through Swin encoder and respective decoders.

        Args:
            images: List[[B, 3, H, W]]
            points: Keypoints per graph
            graphs: List of Graph objects
            n_points: List[int]
            n_points_sample: List[int]
            perm_mats: [n x n] list of [B, K, K] permutation matrices

        Returns:
            decoded_graphs: List[[B, K, D]]
            attention_maps: List[Dict]
            similarity_matrices: List[List[[B, K, K]]]
        """
         
        batch_size = graphs[0].num_graphs
        orig_graph_list = []
        encoded_graphs, masks = [], []
        # for visualisation purposes only
        graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
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
            global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            
            # global_feature = self.linear_cls(global_feature)
            
            h_res = torch.cat([global_feature, h_res], dim=1)

            global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
            mask = torch.cat([global_feature_mask, mask], dim=1)
            
            # apply mask to global token if active and padding of keypoints
            for b in range(h_res.size(0)):
                cutoff = int(n_points_sample[b])
                if cfg.Matching_TF.global_feat:
                    mask[b, cutoff + 1:] = False
                    h_res[b, cutoff + 1:, :] = 0
                else:
                    mask[b, cutoff:] = False
                    h_res[b, cutoff:, :] = 0
            
            encoded_graphs.append(h_res)
            masks.append(mask)

            orig_graph_list.append((h_res,mask))

        decoded_graphs = []

        for i in range(self.num_graphs):
            decoder = self.graph_decoders[i]
                
            memory = torch.cat([encoded_graphs[j] for j in range(self.num_graphs) if j != i], dim=1) # all graphs not i [B, (G-1) * L, D]
            memory_mask = torch.cat([masks[j] for j in range(self.num_graphs) if j != i], dim=1)    # all masks not i:  unsqueeze each to [B, G-1, L]

            target = encoded_graphs[i]  # current graph encoded feature
            
            #print("DECODER called with")
            #print("---------------------------------------------")
            #print(f"source graph encoding: {target.shape}")
            #print(f"mask: None")
            #print(f"padding mask: {memory_mask.shape}")
            #print(f"encoder_output: {memory.shape}")
            #print("---------------------------------------------")
            
            # no autoregressive mask
            dec_out = decoder(
                source_nodes=target,
                mask=None,
                padding_mask=memory_mask,
                encoder_output=memory,
                is_eval=not in_training
            )
            dec_out = dec_out[:, 1:, :] # [B, K, D]
            decoded_graphs.append(dec_out)
        
        # Store pairwise cosine similarity for future visualization
        self.similarity_matrices = [] # reset List
        for i in range(self.num_graphs):
            row = []
            for j in range(self.num_graphs):
                sim = self.w_cosine(decoded_graphs[i], decoded_graphs[j])
                row.append(sim.detach().cpu())
            self.similarity_matrices.append(row)

        return decoded_graphs, self.similarity_matrices

class PairwiseWeightedCosineSimilarity(nn.Module):
    def __init__(self, node_feature_dim):
        super(PairwiseWeightedCosineSimilarity, self).__init__()
        # Initialize weights with ones for each feature dimension
        self.w = nn.Parameter(torch.ones(1, 1, node_feature_dim))
    
    def forward(self, x, y):
        # x and y have shape [batch_size, nodes, node_feature]
        
        # Apply weights
        x_weighted = x #* self.w  # Shape: [batch_size, nodes_x, node_feature]
        y_weighted = y #* self.w  # Shape: [batch_size, nodes_y, node_feature]
        
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