import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from graph_rewiring import KNN, add_edges, edge_sampling, GDCWrapper
from utils import DummyData, get_full_adjacency
from function_transformer_attention import SpGraphTransAttentionLayer
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc


class GrandDiscritizedBlock(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(GrandDiscritizedBlock, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt,
                                                          device, edge_weights=self.edge_weight).to(device)

  def multiply_attention(self, x, attention, v=None):
    # todo would be nice if this was more efficient
    if self.opt['mix_features']:
      vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
      ax = self.multihead_att_layer.Wout(vx)
    else:
      mean_attention = attention.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1
    attention, values = self.multihead_att_layer(x, self.edge_index)
    ax = self.multiply_attention(x, attention, values)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * x
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GrandDiscritizedNet(BaseGNN):
  def __init__(self, in_features, out_features, hidden_dim, opt, data, device):
    super(GrandDiscritizedNet, self).__init__(opt, data, device)
    opt["add_source"] = True
    self.mol_list = nn.ModuleList()
    self.mol_list.append(
        GrandDiscritizedBlock(in_features, hidden_dim[0], opt, data, device)
    )
    self.opt = opt
    self.data = data
    self.device = device
    self.data_edge_index = data.data.edge_index.to(device)
    self.fa = get_full_adjacency(self.num_nodes).to(device)
    for id in range(len(hidden_dim) - 1):
      self.mol_list.append(GrandDiscritizedBlock(in_features, hidden_dim[id + 1], opt, data, device))
    self.mol_list.append(GrandDiscritizedBlock(in_features,out_features, opt,data, device))
    ###################################3333
#    self.data_edge_index = dataset.data.edge_index.to(device)
#    self.fa = get_full_adjacency(self.num_nodes).to(device)
  def forward(self, x, pos_encoding):
    out = x
    if self.opt['use_labels']:
        y = x[:, -self.num_classes:]
        x = x[:, :-self.num_classes]

    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
    for i in range(len(self.mol_list)):
      #print(f"After layers number {i+1}")
      out = self.mol_list[i](out)
    return out 
class GrandExtendDiscritizedNet(GrandDiscritizedNet):
  def __init__(self, in_features, out_features, hidden_dim, opt, data, device):
    super().__init__(in_features, out_features, hidden_dim, opt, data, device)
  def forward(self,x):
    out = x
    #print("")
    #print(torch.norm(x, dim=(-1)))

    for i in range(len(self.mol_list)):
      out = self.mol_list[i](out) * torch.norm(x, dim=(-1), keepdim=True)
      print(f"After layers number {i+1}")
    return out


######################################################33
if __name__ == "__main__":
    print(f"Test the grand_discritized file")
    device = "cuda"
    dataset = get_dataset(opt, '../data', False)
    dataset.data = dataset.data.to(device, non_blocking=True)
    print(type(dataset.data.x))
    func = GrandExtendDiscritizedNet(dataset.data.num_features, 6, [20,10,4], opt, dataset.data, device)
    print(type(dataset.data))
    out = func(dataset.data.x)
