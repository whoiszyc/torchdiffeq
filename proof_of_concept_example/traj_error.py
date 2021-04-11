import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time


true_y = pd.read_csv("Wt_non_1.csv")
true_y = true_y.to_numpy()
y_linear = pd.read_csv("Wt_lin_1.csv")
y_linear = y_linear.to_numpy()
y_reduced = pd.read_csv("Wt_sma_1.csv")
y_reduced = y_reduced.to_numpy()

error_linear = np.mean(np.absolute(true_y - y_linear))
error_reduced = np.mean(np.absolute(true_y - y_reduced))