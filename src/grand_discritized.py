import torch
from torch import nn
import torch.nn.functional as F
# from graph_rewiring import KNN, add_edges, edge_sampling, GDCWrapper
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from utils import DummyData, get_full_adjacency
from function_transformer_attention import SpGraphTransAttentionLayer
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc
import wandb


class GrandDiscritizedBlock(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(GrandDiscritizedBlock, self).__init__(opt, data, device)
    data = data.data
    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    self.edge_index = self.edge_index.to(device)
    try:
        self.edge_weight = self.edge_weight.to(device)
    except Exception as e:
        pass
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
  def __init__(self, hidden_dim, opt, data, device):
    super(GrandDiscritizedNet, self).__init__(opt, data, device)
#    opt["add_source"] = True
    self.mol_list = nn.ModuleList()
    self.mol_list.append(
        GrandDiscritizedBlock(opt["hidden_dim"], hidden_dim, opt, data, device).to(device)
    )
    self.opt = opt
    self.data = data.data
    self.device = device
    self.data_edge_index = data.data.edge_index.to(device)
    self.fa = get_full_adjacency(self.num_nodes).to(device)
    for id in range(opt["depth"]):
      self.mol_list.append(GrandDiscritizedBlock(opt["hidden_dim"], hidden_dim, opt, data, device).to(device))
    self.mol_list.append(GrandDiscritizedBlock(opt["hidden_dim"], hidden_dim, opt,data, device).to(device))
    ###################################3333
#    self.data_edge_index = dataset.data.edge_index.to(device)
#    self.fa = get_full_adjacency(self.num_nodes).to(device)
    def forward(self, x, pos_encoding = None):
    # Encode
      if self.opt['use_labels']:
        y = x[:, -self.num_classes:]
        x = x[:, :-self.num_classes]

      out = x
      for i in range(len(self.mol_list)):
      #print(f"After layers number {i+1}")
        out = self.mol_list[i](out)
      return out 
class GrandExtendDiscritizedNet(GrandDiscritizedNet):
  def __init__(self, opt, data, device):
    super().__init__(opt["hidden_dim"], opt, data, device)
    self.discritize_type = opt["discritize_type"]
    self.truncate_tensor = torch.Tensor([opt["truncate_coeff"]]).to(device)
    self.norm_exp = torch.Tensor([opt["norm_exp"]]).to(device)
    if opt['learnable']:
      print(" --> Creating the Custom Parameters\n")
#      _truncate_tensor = nn.Parameter(torch.Tensor([opt["truncate_coeff"]]), requires_grad = True)
#      _norm_exp = nn.Parameter(torch.Tensor([opt["norm_exp"]]), requires_grad = True)
      _step_size = nn.Parameter(torch.Tensor([opt["step_size"]]), requires_grad = True)
#      self.register_parameter("truncate_tensor",_truncate_tensor)
#      self.register_parameter("norm_exp", _norm_exp)
      self.register_parameter("step_size", _step_size)
      """
      self.norm_exp = self.norm_exp.to(device)
      self.step_size = self.step_size.to(device)
      self.truncate_tensor = self.truncate_tensor.to(device)
      """
      print(self.norm_exp, self.step_size, self.truncate_tensor, sep="\n\n\n")
      print(" --> Parameter was initialize can be learn\n")
      print(f"{self.truncate_tensor}\n{self.norm_exp}\n{self.step_size}\n")
    else:
      self.step_size = torch.Tensor([opt["step_size"]]).to(device)
    self.opt = opt
  def forward(self,x, pos_encoding=False):
#    print(x.shape, " this is shape before doing anything")
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]
    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
      p = self.mp(p)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)
    out = x
#    print(f"This is the output shape before forward those Blocks: {x.shape}")
    for i in range(len(self.mol_list)):
      if self.opt['discritize_type']=="norm":
        if self.opt['truncate_norm']:	
          out = out + self.step_size * self.mol_list[i](out) * torch.minimum(torch.norm(out, dim=(-1), keepdim=True)**self.norm_exp, self.truncate_tensor)
        else:
          out = out + self.step_size * self.mol_list[i](out) * torch.norm(out, dim=(-1), keepdim=True)**self.norm_exp			
        ####
		
      elif self.discritize_type == "frobenius_norm":
        if self.opt['truncate_norm']:
          out = out + self.step_size * self.mol_list[i](out) * torch.minimum(torch.norm(out, keepdim=True)**self.norm_exp, self.truncate_tensor)
        else:
          out = out + self.mol_list[i](out) * self.step_size * torch.norm(out, keepdim=True)**self.norm_exp
      else:
        out = out + self.step_size * self.mol_list[i](out)
#      print(f"After layers number {i+1}")
    z = out
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z

    #print("")
    #print(torch.norm(x, dim=(-1)))
    

class GrandExtendDiscritizedNet_KNN(GrandDiscritizedNet):
  def __init__(self, opt, dataset, device):
    super(GrandExtendDiscritizedNet_KNN).__init__(opt,dataset, device)
  def forward(self, x, pos_encoding):
#    print(x.shape, " this is shape before doing anything")
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]
    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
      p = self.mp(p)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)
    out = x
#    print(f"This is the output shape before forward those Blocks: {x.shape}")
    for i in range(len(self.mol_list)):
      if self.opt['discritize_type']=="norm":
        if self.opt['truncate_norm']:	
          out = out + self.step_size * self.mol_list[i](out) * torch.minimum(torch.norm(out, dim=(-1), keepdim=True)**self.norm_exp, self.truncate)
        else:
          out = out + self.step_size * self.mol_list[i](out) * torch.norm(out, dim=(-1), keepdim=True)**self.norm_exp			
        ####
		
      elif self.discritize_type == "frobenius_norm":
        if self.opt['truncate_norm']:
          out = out + self.step_size * self.mol_list[i](out) * torch.minimum(torch.norm(out, keepdim=True)**self.norm_exp, self.truncate_tensor)
        else:
          out = out + self.mol_list[i](out) * self.step_size * torch.norm(out, keepdim=True)**self.norm_exp
      elif self.discritized_type == "mm_regular":
        out = out + self.step_size * self.mol_list[i](out) * ((out) / torch.sum(out))
      elif self.discritized_type == "softmax":
        out = out + self.step_size * self.mol_list[i](out) * torch.exp(out) / torch.sum(torch.exp(out))
      else:
        out = out + self.step_size * self.mol_list[i](out)
#      print(f"After layers number {i+1}")
    z = out
    if self.opt['fa_layer']:
      #self.edge_index = add_edges(self, self.opt)  
      pass
    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z

######################################################33
if __name__ == "__main__":
  
  print(f"Test the grand_discritized file")
  opt = {'depth': 5,'use_cora_defaults': False, 'dataset': 'Cora', 'data_norm': 'rw', 'self_loop_weight': 1.0, 'use_labels': False, 'geom_gcn_splits': False, 'num_splits': 1, 'label_rate': 0.5, 'planetoid_split': False, 'hidden_dim': 16, 'fc_out': False, 'input_dropout': 0.5, 'dropout': 0.0, 'batch_norm': False, 'optimizer': 'adam', 'lr': 0.01, 'decay': 0.0005, 'epoch': 100, 'alpha': 1.0, 'alpha_dim': 'sc', 'no_alpha_sigmoid': False, 'beta_dim': 'sc', 'block': 'constant', 'function': 'laplacian', 'use_mlp': False, 'add_source': False, 'cgnn': False, 'time': 1.0, 'augment': False, 'method': None, 'step_size': 1, 'max_iters': 100, 'adjoint_method': 'adaptive_heun', 'adjoint': False, 'adjoint_step_size': 1, 'tol_scale': 1.0, 'tol_scale_adjoint': 1.0, 'ode_blocks': 1, 'max_nfe': 1000, 'no_early': False, 'earlystopxT': 3, 'max_test_steps': 100, 'leaky_relu_slope': 0.2, 'attention_dropout': 0.0, 'heads': 4, 'attention_norm_idx': 0, 'attention_dim': 64, 'mix_features': False, 'reweight_attention': False, 'attention_type': 'scaled_dot', 'square_plus': False, 'jacobian_norm2': None, 'total_deriv': None, 'kinetic_energy': None, 'directional_penalty': None, 'not_lcc': True, 'rewiring': None, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_k': 64, 'gdc_threshold': 0.0001, 'gdc_avg_degree': 64, 'ppr_alpha': 0.05, 'heat_time': 3.0, 'att_samp_pct': 1, 'use_flux': False, 'exact': False, 'M_nodes': 64, 'new_edges': 'random', 'sparsify': 'S_hat', 'threshold_type': 'topk_adj', 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'KNN_online': False, 'KNN_online_reps': 4, 'KNN_space': 'pos_distance', 'beltrami': False, 'fa_layer': False, 'pos_enc_type': 'DW64', 'pos_enc_orientation': 'row', 'feat_hidden_dim': 64, 'pos_enc_hidden_dim': 32, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_epoch': 5, 'edge_sampling_add': 0.64, 'edge_sampling_add_type': 'importance', 'edge_sampling_rmv': 0.32, 'edge_sampling_sym': False, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_space': 'attention', 'symmetric_attention': False, 'fa_layer_edge_sampling_rmv': 0.8, 'gpu': 0, 'pos_enc_csv': False, 'pos_dist_quantile': 0.001, 'discritize_type': 'norm'}
  device = "cuda"
  dataset = get_dataset(opt, '../data', False)
  dataset.data = dataset.data.to(device, non_blocking=True)
  print(type(dataset.data.x))
  print(type(dataset.data))
  func = GrandExtendDiscritizedNet(opt, dataset, device).to(device)
  out = func(dataset.data.x)

