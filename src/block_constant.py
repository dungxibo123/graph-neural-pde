from base_classes import ODEblock
import torch
from utils import get_rw_adj
from regularized_ODE_function import RegularizedODEfunc


class ConstantODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, num_nodes, edge_index, edge_attr, device, t=torch.tensor([0, 1])):
    super(ConstantODEblock, self).__init__(odefunc, regularization_fns, opt, t)

    self.aug_dim = 2 if opt['augment'] else 1
    # todo not ideal that the class and object have the same name
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, edge_index, edge_attr, device)
    self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)
    self.odefunc.edge_index, self.odefunc.edge_weight = get_rw_adj(edge_index, edge_weight=edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=num_nodes)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight 

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()

  def forward(self, x):
    t = self.t.type_as(x)

    integrator = self.train_integrator if self.training else self.test_integrator
    
    reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    state = (x,) + reg_states if self.training and self.nreg > 0 else x

    if self.opt["adjoint"]:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        adjoint_method=self.opt['adjoint_method'],
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple( st[1] for st in state_dt[1:] )
      return z, reg_states
    else: 
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
