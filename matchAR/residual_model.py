import numpy as np
import torch
import torch.nn as nn
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


class ResMatcherNet(utils.backbone.VGG16_bn):
    def __init__(self):
        super(ResMatcherNet, self).__init__()
        self.model_name = 'Transformer'
        self.psi = SConv(input_features=cfg.SPLINE_CNN.input_features, output_features=cfg.Matching_TF.d_model)
        self.mlp = MLPQuery(cfg.Matching_TF.d_model, 1024, cfg.Matching_TF.d_model, batch_norm=cfg.Matching_TF.batch_norm)
        
        self.vgg_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(512, cfg.Matching_TF.d_model)

        self.s_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.t_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.cls_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.pos_encoding = Pointwise2DPositionalEncoding(cfg.Matching_TF.d_model, 256, 256).cuda()


        self.transformer = nn.Transformer(d_model= cfg.Matching_TF.d_model,
                                          nhead= cfg.Matching_TF.n_head, 
                                          num_encoder_layers=cfg.Matching_TF.n_encoder, 
                                          num_decoder_layers=cfg.Matching_TF.n_decoder,
                                          batch_first=True, 
                                          activation=cfg.Matching_TF.activation)
        # self.mlp_out = MLP([cfg.Matching_TF.d_model, cfg.Matching_TF.d_model], 1, batch_norm=cfg.Matching_TF.batch_norm)
        self.mlp_out = MLP([cfg.Matching_TF.d_model, 512, 1024, 512, 256], 1, batch_norm=False)
        self.global_state_dim = 1024

    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        perm_mats,
        visualize_flag=False,
        visualization_params=None,
    ):
        batch_size = graphs[0].num_graphs
        global_list = []
        orig_graph_list = []
        node_feat_list = []
        # for visualisation purposes only
        graph_list = []
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
            
            if cfg.Matching_TF.global_feat:
                # with torch.no_grad():
                global_feature = self.final_layers(edges)[0].reshape((nodes.shape[0], -1))
                global_feature = self.glob_to_node_dim(global_feature)
                global_feature = global_feature + self.cls_enc
                global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
                h_res = torch.cat([h_res, global_feature], dim=1)

                global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
                mask = torch.cat([mask, global_feature_mask], dim=1)


            orig_graph_list.append((h_res,mask))

        h_s, s_mask = orig_graph_list[0]
        h_t, t_mask = orig_graph_list[1]

        assert h_s.size(0) == h_t.size(0), 'batch-sizes are not equal'
        (B, N_s, D), N_t = h_s.size(), h_t.size(1)
        
        S_mask = ~torch.cat((s_mask, t_mask), dim=1)

        if cfg.Matching_TF.global_feat != True:
            (B, N_s, D), N_t = h_s.size(), h_t.size(1)
            query_mask = ~(s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t))
            queries = make_queries(h_s, h_t)

        else:
            (B, N_s, D), N_t = h_s.size(), h_t.size(1)
            N_s -= 1
            N_t -= 1 
            query_mask = ~(s_mask[:,:-1].view(B, N_s, 1) & t_mask[:,:-1].view(B, 1, N_t))
            queries = make_queries(h_s[:,:-1,:], h_t[:,:-1,:])
        
        query_mask = query_mask.view(B, -1)
        
        input = torch.cat((h_s + self.s_enc, h_t + self.t_enc), dim=1)
        # print('in_transformer: ', input.size())
        # print('queries: ', queries.size())
        queries = self.mlp(queries)
        # print('mlp out: ', queries.size())
        transformer_out = self.transformer(input, 
                                  queries, 
                                  src_key_padding_mask= S_mask,
                                  memory_key_padding_mask= S_mask,
                                  tgt_key_padding_mask= query_mask)
        # print('out_transformer: ', transformer_out.size())
        output = self.mlp_out(transformer_out)
        # print('output: ', output.size())

        # C = - output.view(batch_size, N_s, N_t)
        # y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
        #                     for x in range(batch_size)])).to(output.device)
        # matchings = [make_perm_mat_pred(y_pred[:,1,:], N_t).to(output.device)]

        # if visualize_flag:
        #     easy_visualize(
        #         graph_list,
        #         points,
        #         n_points,
        #         images,
        #         matchings,
        #         **visualization_params,
        #     )

        return output.squeeze(2)


class MLP(nn.Module):
    def __init__(self, h_sizes, out_size, batch_norm):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.batch_norm = batch_norm
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(h_sizes[k+1]))
        self.out = nn.Linear(h_sizes[-1], out_size)
    
    def forward(self, x):

        # Feedforward
        for i in range(len(self.hidden)):
            if self.batch_norm:
                x = torch.transpose(self.bn_layers[i](torch.transpose(self.hidden[i](x),1,2)),1,2)
            else:
                    x = self.hidden[i](x)
            x = nn.functional.relu(x)
        output = self.out(x)
        return output

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
