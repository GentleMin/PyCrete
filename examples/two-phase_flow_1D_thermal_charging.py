"""
Code for Project A, Task 1, SciDL 2023
Jingtao Min @ EPM, ETH Zurich
"""

import numpy as np
import torch
import pandas as pd
import pinn_core

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PiNN_2Phase_Thermal_2Nets(pinn_core.PiNN_Cartesian):

    def __init__(self, forward_approx_f, forward_approx_s, loss_weights, *args,
                 diffuse_f=0.05, diffuse_s=0.08, couple_f=5., couple_s=6., background_flow=1., T0=1., Th=4.,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.approximator_f = forward_approx_f
        self.approximator_s = forward_approx_s
        self.wt = loss_weights
        self.diffuse_f = diffuse_f
        self.diffuse_s = diffuse_s
        self.couple_f = couple_f
        self.couple_s = couple_s
        self.U0 = background_flow
        self.bound_f, self.bound_s = self.boundary_condition(T0, Th)

    def boundary_condition(self, T0, Th):
        bound_f = [
            [T0*torch.ones(self.data_boundary[0][0].shape[0]).to(device), ],
            [T0 + (Th - T0)/(1. + torch.exp(-200.*(self.data_boundary[1][0][:, 0] - 0.25))),
             torch.zeros(self.data_boundary[1][1].shape[0]).to(device)]
        ]
        bound_s = [
            [T0*torch.ones(self.data_boundary[0][0].shape[0]).to(device),],
            [torch.zeros(self.data_boundary[1][0].shape[0]).to(device),
             torch.zeros(self.data_boundary[1][1].shape[0]).to(device)]
        ]
        return bound_f, bound_s

    def forward(self, x_input):
        return 3.*self.approximator_f(self.normalize(x_input)) + 1., 3.*self.approximator_s(self.normalize(x_input)) + 1.

    def residual(self, x_int):
        x_int.requires_grad = True
        Tf, Ts = self.forward(x_input=x_int)

        grad_Tf = torch.autograd.grad(Tf.sum(), x_int, create_graph=True)[0]
        dTf_dt = grad_Tf[:, 0]
        dTf_dx = grad_Tf[:, 1]
        d2Tf_dx2 = torch.autograd.grad(dTf_dx.sum(), x_int, create_graph=True)[0][:, 1]

        grad_Ts = torch.autograd.grad(Ts.sum(), x_int, create_graph=True)[0]
        dTs_dt = grad_Ts[:, 0]
        dTs_dx = grad_Ts[:, 1]
        d2Ts_dx2 = torch.autograd.grad(dTs_dx.sum(), x_int, create_graph=True)[0][:, 1]

        res_Tf = dTf_dt + self.U0*dTf_dx - self.diffuse_f*d2Tf_dx2 + self.couple_f*(Tf - Ts).squeeze()
        res_Ts = dTs_dt - self.diffuse_s*d2Ts_dx2 - self.couple_s*(Tf - Ts).squeeze()

        return res_Tf, res_Ts

    def boundary_loss(self):

        Tf_init, Ts_init = self.forward(self.data_boundary[0][0])
        loss_Tf_init = torch.mean(torch.abs(Tf_init.squeeze() - self.bound_f[0][0])**2)
        loss_Ts_init = torch.mean(torch.abs(Ts_init.squeeze() - self.bound_s[0][0])**2)

        self.data_boundary[1][0].requires_grad = True
        Tf_lbound, Ts_lbound = self.forward(self.data_boundary[1][0])
        dTs_dx_l = torch.autograd.grad(Ts_lbound.sum(), self.data_boundary[1][0], create_graph=True)[0][:, 1]
        loss_Tf_left = torch.mean(torch.abs(Tf_lbound.squeeze() - self.bound_f[1][0])**2)
        loss_Ts_left = torch.mean(torch.abs(dTs_dx_l - self.bound_s[1][0])**2)

        self.data_boundary[1][1].requires_grad = True
        Tf_rbound, Ts_rbound = self.forward(self.data_boundary[1][1])
        dTf_dx_r = torch.autograd.grad(Tf_rbound.sum(), self.data_boundary[1][1], create_graph=True)[0][:, 1]
        dTs_dx_r = torch.autograd.grad(Ts_rbound.sum(), self.data_boundary[1][1], create_graph=True)[0][:, 1]
        loss_Tf_right = torch.mean(torch.abs(dTf_dx_r - self.bound_f[1][1])**2)
        loss_Ts_right = torch.mean(torch.abs(dTs_dx_r - self.bound_s[1][1])**2)

        loss_bound_Tf = [[loss_Tf_init, ], [loss_Tf_left, loss_Tf_right]]
        loss_bound_Ts = [[loss_Ts_init, ], [loss_Ts_left, loss_Ts_right]]

        return loss_bound_Tf, loss_bound_Ts

    def compute_loss(self):
        res_pde = self.residual(self.data_interior)
        loss_Tf_pde, loss_Ts_pde = torch.mean(torch.abs(res_pde[0])**3), torch.mean(torch.abs(res_pde[1])**3)
        loss_Tf_bound, loss_Ts_bound = self.boundary_loss()
        loss = torch.log10(self.wt[0][-1]*loss_Tf_pde
                           + self.wt[0][0][0]*loss_Tf_bound[0][0]
                           + self.wt[0][1][0]*loss_Tf_bound[1][0]
                           + self.wt[0][1][1]*loss_Tf_bound[1][1]
                           + self.wt[1][-1]*loss_Ts_pde
                           + self.wt[1][0][0]*loss_Ts_bound[0][0]
                           + self.wt[1][1][0]*loss_Ts_bound[1][0]
                           + self.wt[1][1][1]*loss_Ts_bound[1][1])
        return loss, loss_Tf_pde, loss_Ts_pde, loss_Tf_bound, loss_Ts_bound


"""Set up networks"""

torch.manual_seed(42)
# Fluid phase network
nn_phase_f = pinn_core.uniform_MLP(in_dim=2, out_dim=1,
                                   neurons=32, n_hidden_layers=3)
nn_phase_f.apply(pinn_core.linear_init_xavier)
nn_phase_f.to(device)
# Solid phase network
nn_phase_s = pinn_core.uniform_MLP(in_dim=2, out_dim=1,
                                   neurons=32, n_hidden_layers=3)
nn_phase_s.apply(pinn_core.linear_init_xavier)
nn_phase_s.to(device)

pinn_2phase = PiNN_2Phase_Thermal_2Nets(nn_phase_f, nn_phase_s,
                    loss_weights=[
                        [[1.,], [1., 1.], 1.],
                        [[1.,], [1., 1.], 1.]
                    ],
                    domain=torch.tensor([[0., 1.], [0., 1.]]).to(device),
                    n_int=10000,
                    n_bound=[200, 500],
                    idx_bound=[(0,), (0, 1)],
                    couple_f=5.,
                    couple_s=6.,
                    device=device
                )


"""Training"""

n_epochs_pre = 100
verbose_epochs_pre = 100
n_epochs = 1
verbose_epochs = 1

optimizer_pre = torch.optim.Adam(list(nn_phase_f.parameters()) + list(nn_phase_s.parameters()),
                                 lr=1e-3, weight_decay=3.e-5)
optimizer = torch.optim.LBFGS(list(nn_phase_f.parameters()) + list(nn_phase_s.parameters()),
                lr=1.0, max_iter=1000, max_eval=1500,
                history_size=150, line_search_fn="strong_wolfe",
                tolerance_change=1.*np.finfo(float).eps)

hist_loss = list()
hist_Tf_pde = list()
hist_Tf_init = list()
hist_Tf_left = list()
hist_Tf_right = list()
hist_Ts_pde = list()
hist_Ts_init = list()
hist_Ts_left = list()
hist_Ts_right = list()

def closure():
    optimizer.zero_grad()
    loss, loss_Tf_pde, loss_Ts_pde, loss_Tf_bound, loss_Ts_bound = pinn_2phase.compute_loss()
    loss.backward()

    hist_loss.append(loss.detach().cpu().item())
    hist_Tf_pde.append(loss_Tf_pde.detach().cpu().item())
    hist_Tf_init.append(loss_Tf_bound[0][0].detach().cpu().item())
    hist_Tf_left.append(loss_Tf_bound[1][0].detach().cpu().item())
    hist_Tf_right.append(loss_Tf_bound[1][1].detach().cpu().item())
    hist_Ts_pde.append(loss_Ts_pde.detach().cpu().item())
    hist_Ts_init.append(loss_Ts_bound[0][0].detach().cpu().item())
    hist_Ts_left.append(loss_Ts_bound[1][0].detach().cpu().item())
    hist_Ts_right.append(loss_Ts_bound[1][1].detach().cpu().item())
    return loss

print("Gradient-based pre-training...")

epoch_split = [0]
for epoch in range(n_epochs_pre):
    optimizer_pre.step(closure=closure)
    if (epoch + 1) % verbose_epochs_pre == 0:
        print("--------------- Epoch {:d} ---------------".format(epoch + 1))
        print("Average training loss = {:f}".format(np.array(hist_loss[epoch_split[-1]:]).mean()))
        epoch_split.append(len(hist_loss))

print("L-BFGS training...")

for epoch in range(n_epochs):

    optimizer.step(closure=closure)
    pinn_2phase.wt[0][1][0] *= 1.1

    if (epoch + 1) % verbose_epochs == 0:
        print("--------------- Epoch {:d} ---------------".format(epoch + 1))
        print("Average training loss = {:f}".format(np.array(hist_loss[epoch_split[-1]:]).mean()))
        epoch_split.append(len(hist_loss))


"""Outputs"""

# Save model
torch.save(nn_phase_f.state_dict(), "./Task1/net_phase_f.pth")
torch.save(nn_phase_s.state_dict(), "./Task1/net_phase_s.pth")

# Prediction
test_ds = pd.read_csv("./Task1/TestingData.txt", header=0)

test_tensor = torch.from_numpy(test_ds.to_numpy()).to(torch.float32).to(device)
test_output_f, test_output_s = pinn_2phase.forward(test_tensor)

test_ds["tf"] = test_output_f.detach().cpu().numpy()
test_ds["ts"] = test_output_s.detach().cpu().numpy()
test_ds.to_csv("./Task1/Task1_temp.txt", index=False)
