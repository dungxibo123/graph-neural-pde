import torch
import torchsde
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
from block_transformer_attention import AttODEblock
import wandb
class AttSDEblock(AttODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t = torch.tensor([0,1]), gamma = 0.5):
    super(AttSDEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t, gamma)
    self.train_integrator = torchsde.sdeint if opt['adjoint'] else torchsde.sdeint_adjoint
    self.test_intergrator = torchsde.sdeint if opt['adjoint'] else torchsde.sdeint_adjoint
    self.odefunc.noise_type = "general"
    self.reg_odefunc.noise_type = "general"
    self.odefunc.sde_type = "ito"
    self.reg_odefunc.noise_type = "ito"

  def forward(self, x):
    t = self.t.type_as(x)
    self.odefunc.attention_weights = self.get_attention_weights(x)
    self.reg_odefunc.odefunc.attention_weights = self.odefunc.attention_weights
    integrator = self.train_integrator if self.training else self.test_integrator

    reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x
    if self.opt['adjoint'] and self.training:
      state_dt = integrator(
        func, state, t,
        method = 'euler',
        names={"drift": "forward", "diffusion": "forwardg"},
        adjoint_method='euler',
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(func, state, t,
                method='euler',
                names={"drift": "forward", "diffusion": "forwardg"},
                atol=self.atol,
                rtol=self.rtol
                )
    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple(st[1] for st in state_dt[1:])
      return z, reg_states
    else:
      z = state_dt[1]
      return z
class GrandSDE(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GrandSDE, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = AttSDEblock(self.f, self.regularization_fns, opt, dataset.data, device, t=time_tensor).to(device)
  def forward(self, x, pos_encoding = None):
    
    # Encode each node based on its feature.
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

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

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
  opt = {'depth': 5,'use_cora_defaults': False, 'dataset': 'Cora', 'data_norm': 'rw', 'self_loop_weight': 1.0, 'use_labels': False, 'geom_gcn_splits': False, 'num_splits': 1, 'label_rate': 0.5, 'planetoid_split': False, 'hidden_dim': 16, 'fc_out': False, 'input_dropout': 0.5, 'dropout': 0.0, 'batch_norm': False, 'optimizer': 'adam', 'lr': 0.01, 'decay': 0.0005, 'epoch': 100, 'alpha': 1.0, 'alpha_dim': 'sc', 'no_alpha_sigmoid': False, 'beta_dim': 'sc', 'block': 'constant', 'function': 'laplacian', 'use_mlp': False, 'add_source': False, 'cgnn': False, 'time': 1.0, 'augment': False, 'method': None, 'step_size': 1, 'max_iters': 100, 'adjoint_method': 'adaptive_heun', 'adjoint': False, 'adjoint_step_size': 1, 'tol_scale': 1.0, 'tol_scale_adjoint': 1.0, 'ode_blocks': 1, 'max_nfe': 100000000, 'no_early': False, 'earlystopxT': 3, 'max_test_steps': 100, 'leaky_relu_slope': 0.2, 'attention_dropout': 0.0, 'heads': 4, 'attention_norm_idx': 0, 'attention_dim': 64, 'mix_features': False, 'reweight_attention': False, 'attention_type': 'scaled_dot', 'square_plus': False, 'jacobian_norm2': None, 'total_deriv': None, 'kinetic_energy': None, 'directional_penalty': None, 'not_lcc': True, 'rewiring': None, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_k': 64, 'gdc_threshold': 0.0001, 'gdc_avg_degree': 64, 'ppr_alpha': 0.05, 'heat_time': 3.0, 'att_samp_pct': 1, 'use_flux': False, 'exact': False, 'M_nodes': 64, 'new_edges': 'random', 'sparsify': 'S_hat', 'threshold_type': 'topk_adj', 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'KNN_online': False, 'KNN_online_reps': 4, 'KNN_space': 'pos_distance', 'beltrami': False, 'fa_layer': False, 'pos_enc_type': 'DW64', 'pos_enc_orientation': 'row', 'feat_hidden_dim': 64, 'pos_enc_hidden_dim': 32, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_epoch': 5, 'edge_sampling_add': 0.64, 'edge_sampling_add_type': 'importance', 'edge_sampling_rmv': 0.32, 'edge_sampling_sym': False, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_space': 'attention', 'symmetric_attention': False, 'fa_layer_edge_sampling_rmv': 0.8, 'gpu': 0, 'pos_enc_csv': False, 'pos_dist_quantile': 0.001, 'discritize_type': 'norm'}
  device = "cuda"
  dataset = get_dataset(opt, '../data', False)
  dataset.data = dataset.data.to(device, non_blocking=True)
  print(type(dataset.data.x))
  print(type(dataset.data))
  func = GrandSDE(opt, dataset, device).to(device)
  out = func(dataset.data.x)

