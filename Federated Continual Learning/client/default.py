# import all libraries in the code below
import flwr as fl
import sys

sys.path.append('..')
from CL.default import Memory
import numpy as np
import pickle
import copy
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
# import ordereddict
from collections import OrderedDict
import flwr as fl
import sys
from utils import get_parameters, set_parameters, train, test, predict, predict_gen

class FlowerClient(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, testloader, epochs, y_labels, num_clients, DEVICE, path):
		self.cid = cid
		self.net = net
		self.trainloader = trainloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.path = path
		self.num_clients = num_clients
		self.DEVICE = DEVICE
	
	def get_parameters(self, config) -> List[np.ndarray]:
		# print(f"[Client {self.cid}] get_parameters")
		self.net.train()
		return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.net.train()
		params_dict = zip(self.net.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		self.set_parameters(parameters)
		train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		return self.get_parameters(config={}), len(self.trainloader), {}
	
	def evaluate(self, parameters, config):
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		self.set_parameters(parameters)
		loss, avg_pearson, avg_rmse = test(self.net, self.testloader, self.y_labels, DEVICE=self.DEVICE)
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClientCL(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
	             strat_name, params):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
	
	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		# self.net.train()
		self.strat.model.train()
		return [val.cpu().numpy() for _, val in self.strat.model.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		# self.net.train()
		print(f"[Client {self.cid}] set_parameters")
		self.strat.model.train()
		params_dict = zip(self.strat.model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.load_state_dict(state_dict, strict=True)
		
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		self.set_parameters(parameters)
		# self.strat.load_model(parameters)
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
		except:
			task_count = 0
		try:
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				reg_term = pickle.load(f)
		except:
			reg_term = {}
			
		if config["server_round"] < int(self.nrounds / 2):
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term,
			#                                                 train_loader=self.trainloaders[0][int(self.cid)], learn_ewc=False))
			self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term, train_loader=self.trainloaders[0][int(self.cid)],
			                       learn_ewc=False)
			# reg_term = self.strat.regularization_terms
			
		elif config["server_round"] == int(self.nrounds / 2):
			task_count += 1
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term,
			#                                                 train_loader=self.trainloaders[0][int(self.cid)], learn_ewc=True))
			
			self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term, train_loader=self.trainloaders[0][int(self.cid)],
			                       learn_ewc=True)
			
			reg_term = self.strat.regularization_terms
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(reg_term, f)
			with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
				f.write(f'{task_count}')
		else:
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term,
			#                                                 train_loader=self.trainloaders[1][int(self.cid)], learn_ewc=False))
			
			self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term, train_loader=self.trainloaders[1][int(self.cid)],
			                       learn_ewc=False)
			# reg_term = self.strat.regularization_terms
		
		
		if config['server_round'] <= int(self.nrounds / 2):
			return self.get_parameters(config={}), len(self.trainloaders[0][int(self.cid)]), {}
		
		return self.get_parameters(config={}), len(self.trainloaders[1][int(self.cid)]), {}
	
	def evaluate(self, parameters, config):
		# check if {self.path}/{self.num_clients} exists, if not create it
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		
		self.set_parameters(parameters)
		# self.strat.load_model(parameters)
		if config["server_round"] <= int(self.nrounds / 2):
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.valloader[0], self.y_labels, self.DEVICE)
			# loss, avg_pearson, avg_rmse = self.strat.test(self.valloader[0])
		else:
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.testloader, self.y_labels, self.DEVICE)
			# loss, avg_pearson, avg_rmse = self.strat.test(self.testloader)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(self.valloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClient_NR(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
	             strat_name, params):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
	
	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		# self.net.train()
		self.strat.model.train()
		return [val.cpu().numpy() for _, val in self.strat.model.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		print(f"[Client {self.cid}] set_parameters")
		# self.net.train()
		self.strat.model.train()
		params_dict = zip(self.strat.model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
		except:
			task_count = 0
		
		if config["server_round"] < int(self.nrounds / 2):
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], learn_nr=False))
			self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], learn_nr=False)
			memory = self.strat.task_memory
		
		elif config["server_round"] == int(self.nrounds / 2):
			print("  \n")
			task_count += 1
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], learn_nr=True))
			self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], learn_nr=True)
			
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(self.strat.task_memory[task_count].storage, f)
		else:
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				stor = pickle.load(f)
			memory = {task_count: Memory()}
			memory[task_count].update(stor)
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, memory, self.trainloaders[1][int(self.cid)], learn_nr=False))
			self.strat.learn_batch(task_count, memory, self.trainloaders[1][int(self.cid)], learn_nr=False)
			memory = self.strat.task_memory
		
		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		if config['server_round'] <= int(self.nrounds / 2):
			return self.get_parameters(config={}), len(self.trainloaders[0][int(self.cid)]), {}
		return self.get_parameters(config={}), len(self.trainloaders[1][int(self.cid)]), {}
	
	def evaluate(self, parameters, config):
		# check if {self.path}/{self.num_clients} exists, if not create it
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		
		# set_parameters(self.strat.model, parameters)
		self.set_parameters(parameters)
		if config["server_round"] <= int(self.nrounds / 2):
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.valloader[0], self.y_labels, self.DEVICE)
		else:
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.testloader, self.y_labels, self.DEVICE)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(self.valloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}