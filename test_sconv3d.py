import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from model.sconv_archs import SConv

device  = 'cuda'
d_model = 648
num_graphs, nodes_per = 2, 4
N = num_graphs * nodes_per

# 1) toy node features
x = torch.randn(N, d_model, device=device)

# 2) toy batch / edge_index for two 4-cycles
batch = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per)
edges = []
for g in range(num_graphs):
    off = g * nodes_per
    for (u,v) in [(0,1),(1,2),(2,3),(3,0)]:
        edges.append([off+u, off+v])
edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
E = edge_index.size(1)

# 3) pseudo‐coordinates.  **Must** live in [-1,1] if you’re going to use SplineConv
#    pipeline’s `build_graphs` will do this for you—
#    here we just sample uniformly in [-1,1]
pseudo3d = torch.rand(E, 3, device=device) * 2 - 1

data_all = Data(x=x,
                edge_index=edge_index,
                edge_attr=pseudo3d,
                batch=batch)

#
# A) Test the 2-D version
#
psi2d = SConv(input_features=d_model,
              output_features=d_model,
              num_layers=2,
              dim=2,           # <- 2D spline!
              kernel_size=5).to(device)

# only keep XY
data2d = Data(x=x,
              edge_index=edge_index,
              edge_attr=pseudo3d[:, :2],
              batch=batch)

out2d = psi2d(data2d)     # should succeed
loss2 = out2d.pow(2).sum()
loss2.backward()
print("✅ 2D SplineConv works")

#
# B) Now bump up to full 3-D
#
psi3d = SConv(input_features=d_model,
              output_features=d_model,
              num_layers=2,
              dim=3,           # <- 3D spline!
              kernel_size=5).to(device)

out3d = psi3d(data_all)   # if this crashes, it’s the 3-D kernel
loss3 = out3d.pow(2).sum()
loss3.backward()
print("✅ 3D SplineConv works")
