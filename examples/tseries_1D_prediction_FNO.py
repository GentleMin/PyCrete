"""
Code for Project A, Task 3, SciDL 2023
Jingtao Min @ EPM, ETH Zurich
"""

import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Arrange data to training sets"""
tseries = pd.read_csv("./Task3/TrainingData.txt", header=0)

len_tseries = tseries.shape[0]
len_i = 34
io_stride = len_i
len_o = 34

np.random.seed(42)
n_samples = 150
idx_start_array = np.random.randint(0, len_tseries - len_i - len_o, size=n_samples)
ds_i = np.zeros((n_samples, len_i, 3))
ds_o = np.zeros((n_samples, len_o, 3))

for i_sample, i_start in enumerate(idx_start_array):
    ds_i[i_sample] = tseries.iloc[i_start:i_start+len_i, :].to_numpy()
    ds_o[i_sample] = tseries.iloc[i_start+io_stride:i_start+io_stride+len_o, :].to_numpy()

np.save("./Task3/tsamples_i.npy", ds_i)
np.save("./Task3/tsamples_o.npy", ds_o)

ds_i = np.load("./Task3/tsamples_i.npy")
ds_o = np.load("./Task3/tsamples_o.npy")


"""Building Fourier Neural Operator"""

class SpectralConv1d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.n_chann_i = in_channels
        self.n_chann_o = out_channels
        self.n_modes = n_modes
        self.scale = 1/(self.n_chann_i*self.n_chann_o)
        self.weights = torch.nn.Parameter(self.scale*torch.rand(self.n_chann_i, self.n_chann_o, self.n_modes, dtype=torch.cfloat))

    def fd_mul(self, inputs, weights):
        return torch.einsum("bix,iox->box", inputs, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        nfft = x.shape[-1]
        x_fd = torch.fft.rfft(x, dim=-1)
        out_fd = torch.zeros(batchsize, self.n_chann_o, nfft // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_fd[:, :, :self.n_modes] = self.fd_mul(x_fd[:, :, :self.n_modes], self.weights)
        out = torch.fft.irfft(out_fd, n=nfft, dim=-1)
        return out

class FourierNeuralOp1d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_modes, width):

        super().__init__()
        self.n_chann_i = in_channels
        self.n_chann_o = out_channels
        self.n_modes = n_modes
        self.width = width

        self.linear_in = torch.nn.Linear(self.n_chann_i, self.width)
        self.fdconv_1 = SpectralConv1d(self.width, self.width, self.n_modes)
        self.lin_1 = torch.nn.Conv1d(self.width, self.width, 1)
        self.fdconv_2 = SpectralConv1d(self.width, self.width, self.n_modes)
        self.lin_2 = torch.nn.Conv1d(self.width, self.width, 1)
        self.fdconv_3 = SpectralConv1d(self.width, self.width, self.n_modes)
        self.lin_3 = torch.nn.Conv1d(self.width, self.width, 1)
        self.linear_merge = torch.nn.Linear(self.width, 32)
        self.linear_out = torch.nn.Linear(32, self.n_chann_o)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def dense_layer(self, x, linear_layer):
        return self.activation(linear_layer(x))

    def forward(self, x):
        x = self.linear_in(x)
        x = x.permute(0, 2, 1)
        x = self.fourier_layer(x, self.fdconv_1, self.lin_1)
        x = self.fourier_layer(x, self.fdconv_2, self.lin_2)
        x = self.fourier_layer(x, self.fdconv_3, self.lin_3)
        x = x.permute(0, 2, 1)
        x = self.dense_layer(x, self.linear_merge)
        x = self.linear_out(x)
        return x


"""Form training datasets"""

temp_min = tseries.iloc[:, 1:].min().min()
temp_max = tseries.iloc[:, 1:].max().max()
t_len = np.ceil(ds_i[0, -1, 0] - ds_i[0, 0, 0])

x_data = torch.from_numpy(ds_i).type(torch.float32).clone()
y_data = torch.from_numpy(ds_o[:, :, 1:]).type(torch.float32).clone()

def normalize_therm(inputs):
    # return (inputs - temp_min)/(temp_max - temp_min)
    return (temp_max - inputs)/(temp_max - temp_min)

def denormalize_therm(inputs):
    # return temp_min + inputs*(temp_max - temp_min)
    return temp_max - inputs*(temp_max - temp_min)

x_data[:, :, 0] = (x_data[:, :, 0].T - x_data[:, 0, 0]).T/t_len
x_data[:, :, 1:] = normalize_therm(x_data[:, :, 1:])
y_data = normalize_therm(y_data)

n_train = 100

input_function_train = x_data[:n_train, :]
output_function_train = y_data[:n_train, :]
input_function_test = x_data[n_train:, :]
output_function_test = y_data[n_train:, :]

batch_size = 10

training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=batch_size, shuffle=True)
testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=batch_size, shuffle=False)


"""Training"""

learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes = 17
width = 64

fno = FourierNeuralOp1d(3, 2, modes, width)

optimizer = torch.optim.Adam(fno.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# refine_epochs = 5
# optimizer_refine = torch.optim.LBFGS(fno.parameters(),
#                                      lr=0.5, max_iter=500, history_size=150,
#                                      line_search_fn="strong_wolfe")

l_mse = torch.nn.MSELoss()

freq_print = 50
for epoch in range(epochs):
    train_mse = 0.0
    for step, (input_batch, output_batch) in enumerate(training_set):
        optimizer.zero_grad()
        output_pred_batch = fno(input_batch).squeeze(2)
        loss_f = l_mse(output_pred_batch, output_batch)
        loss_f.backward()
        optimizer.step()
        train_mse += loss_f.item()
    train_mse /= len(training_set)

    scheduler.step()

    with torch.no_grad():
        fno.eval()
        test_mse = 0.0
        test_relative_l2 = 0.0
        for step, (input_batch, output_batch) in enumerate(testing_set):
            output_pred_batch = fno(input_batch).squeeze(2)
            loss_f = l_mse(output_pred_batch, output_batch)
            test_mse += loss_f.item()
            loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
            test_relative_l2 += loss_f.item()
        test_mse /= len(testing_set)
        test_relative_l2 /= len(testing_set)

    if epoch % freq_print == 0:
        print("------------ Epoch: {:4d} ------ Train loss: {:.5f} ------ Test loss: {:.5f} ------ Relative L2 misfit: {:6.2f}".format(epoch+1, train_mse, test_mse, test_relative_l2))

# L-BFGS seems unable to work on complex layers, see
# https://stackoverflow.com/questions/74574823/can-pytorch-l-bfgs-be-used-to-optimize-a-complex-parameter
# One has to modify the dot product in pytorch source file `lbfgs.py` to allow for correctly handle
# intermediate complex derivatives, which cannot be done on CoLab anyway

# print("L-BFGS refinement...")

# def closure():
#     optimizer_refine.zero_grad()
#     output_pred = fno(input_function_train)
#     loss = l_mse(output_pred, output_function_train)
#     loss.backward()
#     return loss

# for refine_epoch in range(refine_epochs):
#     train_mse = optimizer_refine.step(closure=closure).item()

#     with torch.no_grad():
#         fno.eval()
#         output_pred = fno(input_function_test)
#         test_mse = l_mse(output_pred, output_function_test).item()

#     if refine_epoch % freq_print == 0:
#         print("------------ Epoch: {:4d} ------ Train loss: {:.5f} ------ Test loss: {:.5f}".format(refine_epoch+1, train_mse, test_mse))


"""Prediction"""

dt = tseries["t"][1] - tseries["t"][0]
ensemble_idx = list(range(tseries.shape[0] - 68 + 1, tseries.shape[0] - 34 + 1))
# ensemble_idx = list(range(tseries.shape[0] - 34, tseries.shape[0] - 33))
input_series = np.array([tseries.iloc[idx_start:idx_start+34, :] for idx_start in ensemble_idx])
input_series = torch.from_numpy(input_series).type(torch.float32)
input_series[:, :, 0] = (input_series[:, :, 0].T - input_series[:, 0, 0]).T/t_len
input_series[:, :, 1:] = normalize_therm(input_series[:, :, 1:])

fno.eval()
output_series = fno(input_series)

predicted_t = np.zeros((34, 2))
for idx_pred in range(34):
    idx_batch = torch.arange(idx_pred, 34, dtype=torch.int64)
    idx_pos = 34 + idx_pred - 1 - idx_batch
    predicted_t[idx_pred, :] = output_series[idx_batch, idx_pos, :].mean(dim=0).squeeze().detach().cpu().numpy()

pred_ds = pd.read_csv("./Task3/TestingData.txt", header=0)
pred_ds["tf0"] = denormalize_therm(predicted_t[:, 0])
pred_ds["ts0"] = denormalize_therm(predicted_t[:, 1])
pred_ds.to_csv("./Task3/Task3_m17.txt", index=False)
