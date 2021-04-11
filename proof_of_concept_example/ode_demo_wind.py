import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=190001)
parser.add_argument('--batch_time', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=1000)
parser.add_argument('--viz', action='store_true', default=False)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
true_y = pd.read_csv("Wt_non_1.csv")
true_y = true_y.to_numpy()
true_y = true_y.reshape(190001, 1, 1)
true_y = torch.tensor(true_y)
true_y0 = torch.tensor([[0.0]])
t = torch.linspace(0., 19., args.data_size)

# true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

# class Lambda(nn.Module):
#
#     def forward(self, t, y):
#         return torch.mm(y**3, true_A)


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 4), facecolor='white')
    ax_traj = fig.add_subplot(111, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-0.4, 0.02)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 80),
            nn.Tanh(),
            nn.Linear(80, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    stime = time.time()

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        batch_y0 = batch_y0.float()
        batch_t = batch_t.float()
        batch_y = batch_y.float()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            print("Complete {} training iterations using {}".format(args.test_freq, time.time() - stime))
            stime = time.time()
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y, pred_y, func, ii)
                ii += 1
            print("Complete testing using {}".format(time.time() - stime))
            print("==============================================")
            stime = time.time()
        end = time.time()


pred_y = odeint(func, true_y0, t)

linestyle_tuple = [
    ('solid', (0, ())),  # Same as  or '-'
    ('dashdot', 'dashdot'),  # Same as '-.'
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),
    ('dotted', (0, (1, 1))),
    ('densely dotted', (0, (1, 1))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dotted',        (0, (1, 10))),
     ('loosely dashed',        (0, (5, 10))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))
]
line_colors = ["b", "r", "magenta", "lime", "darkorange", "cyan", "y", "purple", "deepskyblue", "navy", "salmon"]
line_markers = [('s', 'square marker'),
                ('p', 'pentagon marker'),
                ('*', 'star marker'),
                ('o', 'circle marker'),
                ('h', 'hexagon1 marker'),
                ('H', 'hexagon2 marker'),
                ('+', 'plus marker'),
                ('x', 'x marker'),
                ('1', 'tri_down marker'),
                ('2', 'tri_up marker'),
                ('3', 'tri_left marker'),
                ('4', 'tri_right marker'),
                ('8', 'not sure'),
                ('v', 'triangle_down marker'),
                ('^', 'triangle_up marker'),
                ('<', 'triangle_left marker'),
                ('>', 'triangle_right marker'),
                ('D', 'diamond marker'),
                ('d', 'thin_diamond marker'),
                ('|', 'vline marker'),
                ('_', 'hline marker'),
                ('.', 'point marker',),
                (',', 'pixel marker', )]

y_linear = pd.read_csv("Wt_lin_1.csv")
y_linear = y_linear.to_numpy()
y_reduced = pd.read_csv("Wt_sma_1.csv")
y_reduced = y_reduced.to_numpy()

import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Arial'})
fig = plt.figure(figsize=(9, 7), facecolor='white')
plt.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], color='g', label="Nonlinear full-order model", linestyle='solid')
plt.plot(t.cpu().numpy(), y_linear, color='r', label="Linearized full-order model", linestyle='dashed')
plt.plot(t.cpu().numpy(), y_reduced, color='darkorange', label="Linear reduced 1st-order model", linestyle=(0, (5, 1)))
plt.plot(t.cpu().numpy(), pred_y.detach().cpu().numpy()[:, 0, 0], color='b', label="Neural ODE (Nonlinear reduced 1st-order)", linestyle='dashdot')
plt.legend(fontsize=20)
plt.grid()
plt.xlabel("Time (Seconds)", fontsize=20)
plt.ylabel("Wind Turbine Rotor Speed (per unit)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
plt.tight_layout()
plt.savefig("final_comp.png")