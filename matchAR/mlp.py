import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


import utils.backbone
# from BB_GM.affinity_layer import InnerProductWithWeightsAffinity
from matchAR.sconv_archs import SConv
# from lpmp_py import GraphMatchingModule
# from lpmp_py import MultiGraphMatchingModule
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
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


class SimpleNet(utils.backbone.VGG16_bn):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.psi = SConv(input_features=1024, output_features=256)
        self.mlp_proj = MLPQuery(256, 1024, 256, batch_norm=False)

        self.s_enc = nn.Parameter(torch.randn(256))
        self.t_enc = nn.Parameter(torch.randn(256))

        self.mlp = MLP([256, 512, 1024, 512, 256], 1, batch_norm=False)
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

        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            # print('node shape: ',nodes.shape)
            edges = self.edge_layers(nodes)
            # print('edges shape: ',nodes.shape)

            # TODO: Global VGG vector
            # global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            # print('U shape: ',U.shape)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            # print('F shape: ',F.shape)

            node_features = torch.cat((U, F), dim=-1)
            graph.x = node_features
            h = self.psi(graph)
            (h, mask) = to_dense_batch(h, graph.batch, fill_value=0)

            orig_graph_list.append((h,mask))

        h_s, s_mask = orig_graph_list[0]
        h_t, t_mask = orig_graph_list[1]

        assert h_s.size(0) == h_t.size(0), 'batch-sizes are not equal'
        (B, N_s, D), N_t = h_s.size(), h_t.size(1)
        
        S_mask = ~torch.cat((s_mask, t_mask), dim=1)
        query_mask = (s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t))
        query_mask = query_mask.view(B, -1)
        
        queries = make_queries(h_s , h_t)
        queries = self.mlp_proj(queries)
        output = self.mlp(queries).squeeze(2)
        masked_output = torch.where(query_mask, output, float(-10e6))


        return masked_output


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
