from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy
import torch
import torch.optim as optim
import torch.nn as nn
import math
import numpy as np
from scipy.stats import pearsonr
from dataloader.outloader import CustomOutputDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid")

import flwr as fl
from flwr.common import NDArrays, Scalar


class RMSELoss(torch.nn.Module):
	def __init__(self):
		super(RMSELoss, self).__init__()
	
	def forward(self, x, y):
		criterion = nn.MSELoss()
		loss = torch.sqrt(criterion(x, y))
		return loss


def truncate_float(float_number, decimal_places):
	multiplier = 10 ** decimal_places
	return int(float_number * multiplier) / multiplier


def plot_results(history, save_path):
	plt.figure(figsize=(20, 10))
	plt.subplot(1, 3, 1)
	plt.ylim((0, numpy.max([i[0] for i in history.losses_distributed])))
	plt.plot([i[0] for i in history.losses_distributed], [i[1] for i in history.losses_distributed])
	plt.xlabel("Round")
	plt.ylabel("Loss")
	plt.grid(True)
	plt.tight_layout()
	plt.subplot(1, 3, 2)
	plt.ylim((0, numpy.max([i[0] for i in history.metrics_distributed['avg_rmse']])))
	plt.plot([i[0] for i in history.metrics_distributed['avg_rmse']],
	         [i[1] for i in history.metrics_distributed['avg_rmse']])
	plt.xlabel("Round")
	plt.ylabel("RMSE")
	plt.grid(True)
	plt.tight_layout()
	plt.subplot(1, 3, 3)
	plt.ylim((0, 1))
	plt.plot([i[0] for i in history.metrics_distributed['avg_pearson_score']],
	         [i[1] for i in history.metrics_distributed['avg_pearson_score']])
	plt.xlabel("Round")
	plt.ylabel("PCC")
	plt.grid(True)
	plt.tight_layout()
	# output, strategy_name, aug, clients, rounds, epochs = plot_params
	plt.savefig(f"{save_path}_distributed.png")
	# plot centralized results
	try:
		plt.figure(figsize=(20, 10))
		plt.subplot(1, 3, 1)
		plt.ylim((0, numpy.max([i[0] for i in history.losses_centralized])))
		plt.plot([i[0] for i in history.losses_centralized], [i[1] for i in history.losses_centralized])
		plt.xlabel("Round")
		plt.ylabel("Loss")
		plt.grid(True)
		plt.tight_layout()
		plt.subplot(1, 3, 2)
		plt.ylim((0, numpy.max([i[0] for i in history.metrics_centralized['avg_rmse']])))
		plt.plot([i[0] for i in history.metrics_centralized['avg_rmse']],
		         [i[1] for i in history.metrics_centralized['avg_rmse']])
		plt.xlabel("Round")
		plt.ylabel("RMSE")
		plt.grid(True)
		plt.tight_layout()
		plt.subplot(1, 3, 3)
		plt.ylim((0, 1))
		plt.plot([i[0] for i in history.metrics_centralized['avg_pearson_score']],
		         [i[1] for i in history.metrics_centralized['avg_pearson_score']])
		plt.xlabel("Round")
		plt.ylabel("PCC")
		plt.grid(True)
		plt.tight_layout()
		# output, strategy_name, aug, clients, rounds, epochs = plot_params
		plt.savefig(f"{save_path}_centralized.png")
	except:
		print("No centralized results")
	
	# save history as json
	with open(f"{save_path}.txt", "w") as f:
		f.write(str(history.__dict__))


def Average(lst):
	return sum(lst) / len(lst)


def get_parameters(net) -> List[np.ndarray]:
	# use val.gpu for gpu
	net.train()
	return [val.cpu().numpy() for _, val in net.state_dict().items()]


def get_parameters_bn(net) -> List[np.ndarray]:
	# Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
	net.train()
	return [val.cpu().numpy() for name, val in net.state_dict().items() if "bn" not in name]


def set_parameters(net, parameters: List[np.ndarray]):
	net.train()
	params_dict = zip(net.state_dict().keys(), parameters)
	state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=True)
	return net


def set_parameters_bn(net, parameters: List[np.ndarray]):
	net.train()
	keys = [k for k in net.state_dict().keys() if "bn" not in k]
	params_dict = zip(keys, parameters)
	state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=False)
	return net


def train(model, train_loader, epochs, DEVICE):
	criterion = nn.MSELoss()  # Use appropriate loss function based on your task
	# criterion = nn.L1Loss()  # Use appropriate loss function based on your task
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	model.to(DEVICE)
	model.train()
	
	for epoch in range(epochs):
		running_loss = 0.0
		for images, labels in train_loader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def pearson_correlation(labels, outputs):
	# Convert batches of lists to numpy arrays for easier computation
	labels_array = np.array(labels)
	outputs_array = np.array(outputs)
	
	# Calculate mean of labels and outputs along the appropriate axis (usually axis=0 for batches)
	mean_labels = np.mean(labels_array, axis=0)
	mean_outputs = np.mean(outputs_array, axis=0)
	
	# Calculate Pearson correlation coefficient along the appropriate axis
	numerator = np.sum((labels_array - mean_labels) * (outputs_array - mean_outputs), axis=0)
	denominator = np.sqrt(np.sum((labels_array - mean_labels) ** 2, axis=0) * np.sum((outputs_array - mean_outputs) ** 2, axis=0))
	
	# Handling division by zero
	zero_denominator_indices = np.where(denominator == 0)
	correlation = np.zeros_like(denominator)
	correlation[zero_denominator_indices] = 0
	non_zero_indices = np.where(denominator != 0)
	correlation[non_zero_indices] = numerator[non_zero_indices] / denominator[non_zero_indices]
	
	correlation = np.mean(correlation)
	
	return correlation


def test(net, testloader, y_labels, DEVICE):
	criterion = torch.nn.MSELoss()
	RMSE = RMSELoss()
	pearson = {y_labels[i]: [] for i in range(len(y_labels))}
	rmse, total, loss = [], [], []
	net.eval()
	net = net.to(DEVICE)
	with torch.no_grad():
		for features, labels in testloader:
			features, labels = features.to(DEVICE), labels.to(DEVICE)
			outputs = net(features)
			loss.append(criterion(outputs, labels).item())
			rmse.append(RMSE(outputs, labels).item())
			# total += labels.size(0)
			if torch.isnan(features).any():
				print("-----------------------------features")
			if torch.isnan(labels).any():
				print("-----------------------------labels")
			if torch.isnan(outputs).any():
				print("-----------------------------outputs")
			
			labels = labels.cpu().numpy()
			outputs = outputs.cpu().numpy()
			
			for i in range(len(y_labels)):
				if len(labels[:, i]) > 1 and len(outputs[:, i]) > 1:
					temp = pearsonr(labels[:, i], outputs[:, i])[0]
					if math.isnan(temp):
						temp = pearsonr(labels[:, i], np.random.normal(outputs[:, i], 0.0000001))[0]
					pearson[y_labels[i]].append(temp)
	for i in pearson.keys():
		pearson[i] = Average(pearson[i])
	return Average(loss), Average(list(pearson.values())), Average(rmse)


# return loss, pcc, rmse


def predict(net, trainloader, DEVICE, batch_size=16):
	new_pairs = []
	net.eval()  # Set the model to evaluation mode
	net.to(DEVICE)
	with torch.no_grad():
		for inputs, labels in trainloader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			outputs = net(inputs)  # Forward pass through the model
			for i in range(len(outputs)):
				new_pairs.append((outputs[i], labels[i]))
	# batch_size = 1
	new_data = CustomOutputDataset(new_pairs)
	new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
	return new_data_loader


def predict_gen(net, trainloader, DEVICE, batch_size=16):
	new_pairs = []
	net.eval()  # Set the model to evaluation mode
	net.to(DEVICE)
	with torch.no_grad():
		for inputs, labels in trainloader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			outputs = net(inputs)  # Forward pass through the model
			for i in range(len(outputs)):
				new_pairs.append((outputs[i], outputs[i]))
	# batch_size = 1
	new_data = CustomOutputDataset(new_pairs)
	new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
	return new_data_loader


# def predict_gen_distil(net, trainloader):
#     new_pairs = []
#     net.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         for inputs, labels in trainloader:
#             outputs = net(inputs)  # Forward pass through the model
#             new_pairs.append((inputs, outputs))
#     batch_size = 1
#     new_data = CustomOutputDataset(new_pairs) 
#     new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True)
#     return new_data_loader

def predict_gen_distil(net, trainloader, DEVICE, batch_size=16):
	# create new dataframe with predictions of net as labels and images as features keeping in mind that trainloader has batchsize 16
	new_pairs = []
	net.to(DEVICE)
	net.eval()  # Set the model to evaluation mode
	with torch.no_grad():
		for inputs, labels in trainloader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			outputs = net(inputs)
			for i in range(len(outputs)):
				new_pairs.append((inputs[i], outputs[i]))
	new_data = CustomOutputDataset(new_pairs)
	return DataLoader(new_data, batch_size=batch_size, shuffle=True)


def get_eval_fn(net, testloader, y_labels, DEVICE):
	def evaluate(server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
		# print(config.keys())
		net.train()
		params_dict = zip(net.state_dict().keys(), weights)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		net.load_state_dict(state_dict, strict=True)
		loss, avg_pearson, avg_rmse = test(net, testloader, y_labels, DEVICE)
		print("Round %s, Loss %s, Pearson %s, RMSE %s" % (server_round, loss, avg_pearson, avg_rmse))
		return loss, {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}
	
	return evaluate


def get_eval_fn_cl(net, testloader, y_labels, DEVICE):
	def evaluate(server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
		net.train()
		params_dict = zip(net.state_dict().keys(), weights)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		net.load_state_dict(state_dict, strict=True)
		if server_round <= 5:
			loss, avg_pearson, avg_rmse = test(net, testloader[0], y_labels, DEVICE)
		else:
			loss1, avg_pearson1, avg_rmse1 = test(net, testloader[0], y_labels, DEVICE)
			loss2, avg_pearson2, avg_rmse2 = test(net, testloader[1], y_labels, DEVICE)
			loss = (loss1 + loss2) / 2
			avg_pearson = (avg_pearson1 + avg_pearson2) / 2
			avg_rmse = (avg_rmse1 + avg_rmse2) / 2
		
		print("Round %s, Loss %s, Pearson %s, RMSE %s" % (server_round, loss, avg_pearson, avg_rmse))
		return loss, {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}
	
	return evaluate


def get_eval_fn_bn(net, testloader, y_labels, DEVICE):
	def evaluate(server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[
		Tuple[float, Dict[str, Scalar]]]:
		net.train()
		keys = [k for k in net.state_dict().keys() if "bn" not in k]
		params_dict = zip(keys, weights)
		
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		net.load_state_dict(state_dict, strict=False)
		
		loss, avg_pearson, avg_rmse = test(net, testloader, y_labels, DEVICE)
		return loss, {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}
	
	return evaluate


def extract_metrics_gpu_csv(filename):
	df = pd.read_csv(filename)
	# create a new dataframe with RAM, CPU, GPU usage
	df2 = pd.DataFrame(columns=["Strategy", "RAM", "CPU", "GPU_U", "GPU_Mem"])
	for index, row in df.iterrows():
		ram = float(row["RAM After"]) - float(row["RAM Before"])
		ram *= 1024
		cpu_af = list(map(float, row["CPU After"].replace('(', '').replace(')', '').split(', ')))
		cpu_bef = list(map(float, row["CPU Before"].replace('(', '').replace(')', '').split(', ')))
		cpu = cpu_af[0] - cpu_bef[0] + cpu_af[1] - cpu_bef[1]
		# cpu=float(row["CPU After"][0])-float(row["CPU Before"][0]) + float(row["CPU After"][1])-float(row["CPU Before"][1])
		try:
			gpu_af = list(map(float, row["GPU After"].replace('(', '').replace(')', '').split(', ')))
			gpu_bef = list(map(float, row["GPU Before"].replace('(', '').replace(')', '').split(', ')))
			gpu_u = gpu_af[0] - gpu_bef[0]
			gpu_mem = gpu_af[1] - gpu_bef[1]
		except:
			gpu_u = 0
			gpu_mem = 0
		# gpu_u=float(row["GPU After"][0])-float(row["GPU Before"][0])
		# gpu_mem=float(row["GPU After"][1])-float(row["GPU Before"][1])
		df2.loc[index] = [row["Strategy"], ram, cpu, gpu_u, gpu_mem]
	return df2
