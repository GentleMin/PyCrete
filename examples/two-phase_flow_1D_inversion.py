"""
Code for Project A, Task 2, SciDL 2023
Jingtao Min @ EPM, ETH Zurich
"""

import numpy as np
import torch
import pandas as pd
import pinn_core

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Read in observational data (training data)"""

data_train = pd.read_csv("./Task2/DataSolution.txt", header=0)
train_input = torch.from_numpy(data_train.iloc[:, :2].to_numpy()).to(torch.float32).to(device)
train_output = torch.from_numpy(data_train["tf"].to_numpy()).to(torch.float32).to(device)


"""Setup PINN for inverse problem"""

class PiNN_2Phase_Fluid(pinn_core.PiNN_Cartesian):

    def __init__(self, forward_approx_f, forward_approx_s, train_input, train_output, loss_weights, *args,
                 diffuse_f=0.005, couple_f=5., cycle=4., T0=1., Tc=1., Th=4., charge_flow=1., discharge_flow=-1.,
                 charge_range=(0., 1.), discharge_range=(2., 3.), smoothness_reg=0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.approximator_f = forward_approx_f
        self.approximator_s = forward_approx_s
        self.wt = loss_weights
        self.train_i = train_input
        self.train_o = train_output
        self.diffuse_f = diffuse_f
        self.couple_f = couple_f
        self.cycle = cycle
        self.T0 = T0
        self.Tc = Tc
        self.Th = Th
        self.charge_flow = charge_flow
        self.charge_range = charge_range
        self.discharge_flow = discharge_flow
        self.discharge_range = discharge_range
        self.bound_f = self.boundary_condition(T0, Th)
        self.smoothness_reg = smoothness_reg

    def get_background_flow(self, t):
        u0 = torch.zeros(t.shape, device=device)
        cycle_t = t - self.cycle*torch.floor(t/self.cycle)
        u0[(cycle_t >= self.charge_range[0]) & (cycle_t < self.charge_range[1])] = self.charge_flow
        u0[(cycle_t >= self.discharge_range[0]) & (cycle_t < self.discharge_range[1])] = self.discharge_flow
        return u0

    def get_left_bound(self, t):
        bound_left = torch.zeros(t.shape, device=device)
        cycle_t = t - self.cycle*torch.floor(t/self.cycle)
        bound_left[(cycle_t >= self.charge_range[0]) & (cycle_t < self.charge_range[1])] = self.Th
        return bound_left

    def get_right_bound(self, t):
        bound_right = torch.zeros(t.shape, device=device)
        cycle_t = t - self.cycle*torch.floor(t/self.cycle)
        bound_right[(cycle_t >= self.discharge_range[0]) & (cycle_t < self.discharge_range[1])] = self.Tc
        return bound_right

    def boundary_condition(self, T0, Th):
        bound_f = [
            [self.T0*torch.ones(self.data_boundary[0][0].shape[0]).to(device), ],
            [self.get_left_bound(self.data_boundary[1][0][:, 0]),
             self.get_right_bound(self.data_boundary[1][1][:, 0])]
        ]
        return bound_f

    def forward(self, x_input):
        return (self.Th - self.Tc)*self.approximator_f(self.normalize(x_input)) + self.Tc, \
            (self.Th - self.Tc)*self.approximator_s(self.normalize(x_input)) + self.Tc

    def residual(self, x_int):
        x_int.requires_grad = True
        Tf, Ts = self.forward(x_input=x_int)

        grad_Tf = torch.autograd.grad(Tf.sum(), x_int, create_graph=True)[0]
        dTf_dt = grad_Tf[:, 0]
        dTf_dx = grad_Tf[:, 1]
        d2Tf_dx2 = torch.autograd.grad(dTf_dx.sum(), x_int, create_graph=True)[0][:, 1]

        res_Tf = dTf_dt + self.get_background_flow(x_int[:, 0])*dTf_dx - self.diffuse_f*d2Tf_dx2 + self.couple_f*(Tf - Ts).squeeze()
        res_Ts = torch.autograd.grad(Ts.sum(), x_int, create_graph=True)[0][:, 0]

        return res_Tf, res_Ts

    def misfit(self, x_obs, T_out):
        Tf, _ = self.forward(x_input=x_obs)
        return torch.mean(torch.abs(Tf.reshape(-1,) - T_out.reshape(-1,))**2)

    def boundary_loss(self):

        Tf_init, _ = self.forward(self.data_boundary[0][0])
        loss_Tf_init = torch.mean(torch.abs(Tf_init.squeeze() - self.bound_f[0][0])**2)

        self.data_boundary[1][0].requires_grad = True
        Tf_lbound, _ = self.forward(self.data_boundary[1][0])
        dTf_dx_l = torch.autograd.grad(Tf_lbound.sum(), self.data_boundary[1][0], create_graph=True)[0][:, 1]

        cycle_t = self.data_boundary[1][0][:, 0] - self.cycle*torch.floor(self.data_boundary[1][0][:, 0]/self.cycle)
        idx_charge = (cycle_t >= self.charge_range[0]) & (cycle_t < self.charge_range[1])

        res_left = torch.cat([Tf_lbound[idx_charge, 0] - self.bound_f[1][0][idx_charge],
                            #   Tf_lbound[~idx_charge, 0] - self.bound_f[1][0][~idx_charge]])
                              dTf_dx_l[~idx_charge] - self.bound_f[1][0][~idx_charge]])
        loss_Tf_left = torch.mean(torch.abs(res_left)**2)

        self.data_boundary[1][1].requires_grad = True
        Tf_rbound, _ = self.forward(self.data_boundary[1][1])
        dTf_dx_r = torch.autograd.grad(Tf_rbound.sum(), self.data_boundary[1][1], create_graph=True)[0][:, 1]

        cycle_t = self.data_boundary[1][1][:, 0] - self.cycle*torch.floor(self.data_boundary[1][1][:, 0]/self.cycle)
        idx_discharge = (cycle_t >= self.discharge_range[0]) & (cycle_t < self.discharge_range[1])

        res_right = torch.cat([Tf_rbound[idx_discharge, 0] - self.bound_f[1][1][idx_discharge],
                            #    Tf_rbound[~idx_discharge, 0] - self.bound_f[1][1][~idx_discharge]])
                               dTf_dx_r[~idx_discharge] - self.bound_f[1][1][~idx_discharge]])
        loss_Tf_right = torch.mean(torch.abs(res_right)**2)

        loss_bound_Tf = [[loss_Tf_init, ], [loss_Tf_left, loss_Tf_right]]

        return loss_bound_Tf

    def compute_loss(self):
        res_pde, dTs_dt = self.residual(self.data_interior)
        loss_Tf_pde = torch.mean(torch.abs(res_pde)**2)
        loss_Tf_bound = self.boundary_loss()
        loss_train = self.misfit(self.train_i, self.train_o)
        loss_complexity = torch.mean(torch.abs(dTs_dt)**2)
        # loss_train = torch.tensor([0])
        loss = torch.log10(loss_train
                           + self.wt[-1]*loss_Tf_pde
                           + self.wt[0][0]*loss_Tf_bound[0][0]
                           + self.wt[1][0]*loss_Tf_bound[1][0]
                           + self.wt[1][1]*loss_Tf_bound[1][1]
                           + self.smoothness_reg*loss_complexity
                           )
        return loss, loss_train, loss_Tf_pde, loss_Tf_bound, loss_complexity


"""Construct finite basis networks"""

class FiniteBasisPINN(torch.nn.Module):

    def __init__(self, partition_ranges):
        super().__init__()
        self.n_doms = len(partition_ranges)
        self.dom_nets = torch.nn.ModuleList([pinn_core.uniform_MLP(in_dim=2, out_dim=1,
                                                             neurons=16, n_hidden_layers=3) for _ in range(self.n_doms)])
        for dom_net in self.dom_nets:
            dom_net.apply(pinn_core.linear_init_xavier)
        self.dom_nets.to(device)
        self.dom_jacs = [torch.tensor([1./(dom_range[1] - dom_range[0]), 1.]).to(device) for dom_range in partition_ranges]
        self.dom_init = [torch.tensor([dom_range[0], 0.]).to(device) for dom_range in partition_ranges]
        self.dom_cent = [(dom_range[0] + dom_range[1])/2 for dom_range in partition_ranges]

    def forward(self, x):
        y = [torch.exp(-((x[:, 0] - self.dom_cent[i_dom])*self.dom_jacs[i_dom][0])**6/2).unsqueeze(1)* \
             self.dom_nets[i_dom]((x - self.dom_init[i_dom])*self.dom_jacs[i_dom])
            for i_dom in range(self.n_doms)]
        return sum(y)

torch.manual_seed(42)
fbnn_f = FiniteBasisPINN([[-1., -0.5], [-0.5, 0.], [0., 0.5], [0.5, 1.0]])
fbnn_s = FiniteBasisPINN([[-1., -0.5], [-0.5, 0.], [0., 0.5], [0.5, 1.0]])

trange = [0., 8.]
xrange = [0., 1.]

pinn_2phase = PiNN_2Phase_Fluid(fbnn_f, fbnn_s,
                    train_input, train_output,
                    loss_weights=[[1.,], [1., 1.], 1.],
                    domain=torch.tensor([trange, xrange]).to(device),
                    n_int=40000,
                    n_bound=[200, 800],
                    idx_bound=[(0,), (0, 1)],
                    smoothness_reg=0.,
                    bc_weight=[10., 1e-4],
                    device=device
                )


"""Training"""

n_epochs_pre = 2000
verbose_epochs_pre = 100
n_epochs = 20
verbose_epochs = 1

optimizer_pre = torch.optim.Adam(list(fbnn_f.parameters()) + list(fbnn_s.parameters()),
                                 lr=3e-3, weight_decay=1.e-5)
optimizer = torch.optim.LBFGS(list(fbnn_f.parameters()) + list(fbnn_s.parameters()),
                lr=0.7, max_iter=1000, max_eval=1500,
                history_size=150, line_search_fn="strong_wolfe",
                tolerance_change=1.*np.finfo(float).eps)

hist_loss = list()
hist_Tf_misfit = list()
hist_Tf_pde = list()
hist_Tf_init = list()
hist_Tf_left = list()
hist_Tf_right = list()
hist_complexity = list()

def closure():
    optimizer.zero_grad()
    loss, loss_Tf_train, loss_Tf_pde, loss_Tf_bound, loss_complexity = pinn_2phase.compute_loss()
    loss.backward()
    hist_loss.append(loss.detach().cpu().item())
    hist_Tf_misfit.append(loss_Tf_train.detach().cpu().item())
    hist_Tf_pde.append(loss_Tf_pde.detach().cpu().item())
    hist_Tf_init.append(loss_Tf_bound[0][0].detach().cpu().item())
    hist_Tf_left.append(loss_Tf_bound[1][0].detach().cpu().item())
    hist_Tf_right.append(loss_Tf_bound[1][1].detach().cpu().item())
    hist_complexity.append(loss_complexity.detach().cpu().item())
    return loss

print("Gradient-based pre-training...")

epoch_split = [0]
for epoch in range(n_epochs_pre):
    optimizer_pre.step(closure=closure)
    # scheduler_pre.step()
    if (epoch + 1) % verbose_epochs_pre == 0:
        print("--------------- Epoch {:d} ---------------".format(epoch + 1))
        print("Average training loss = {:f}".format(np.array(hist_loss[epoch_split[-1]:]).mean()))
        epoch_split.append(len(hist_loss))

print("L-BFGS training...")

for epoch in range(n_epochs):

    optimizer.step(closure=closure)

    if (epoch + 1) % verbose_epochs == 0:
        print("--------------- Epoch {:d} ---------------".format(epoch + 1))
        print("Average training loss = {:f}".format(np.array(hist_loss[epoch_split[-1]:]).mean()))
        epoch_split.append(len(hist_loss))


"""Output"""

test_ds = pd.read_csv("./Task2/SubExample.txt", header=0)
test_tensor = torch.from_numpy(test_ds.iloc[:, :2].to_numpy()).to(torch.float32).to(device)
test_output_f, test_output_s = pinn_2phase.forward(test_tensor)
test_ds["ts"] = test_output_s.detach().cpu().numpy()
test_ds.to_csv("./Task2/Task2_temp.txt", index=False)
