import argparse

from CL.default import EWC
from client.default import FlowerClientCL, FlowerClient_NR
from dataloader.utils import splitter
from server.strategies import FedAvgWithAccuracyMetric
import ray
import flwr as fl
from server.utils import fit_config, evaluate_config
import torch
import matplotlib.pyplot as plt
from utils import get_parameters, extract_metrics_gpu_csv
import pandas as pd
import os
from CL.default import EWCOnline, SI, MAS
from CL.default import Naive_Rehearsal as NR
from datetime import datetime
from metrics.computation import RAMU, CPUUsage, GPUUsage
from utils import plot_results, get_eval_fn, truncate_float
import gc
import warnings
import numpy as np
import random

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# When running on the CuDNN backend, two further options must be set
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# Set a fixed value for the hash seed
	os.environ["PYTHONHASHSEED"] = str(seed)
	print(f"Random seed set as {seed}")

def run_strategy(strategy, strategy_name, coeff, client_fn, clients, rounds, epochs, output, aug, ray_init_args, client_res):
	print("Running strategy " + str(strategy_name) + " for " + str(clients) + "clients!")
	history = fl.simulation.start_simulation(
		client_fn=client_fn,
		num_clients=int(clients),
		config=fl.server.ServerConfig(num_rounds=int(rounds)),  # Just three rounds
		strategy=strategy,
		ray_init_args=ray_init_args,
		client_resources=client_res,
	)
	try:
		data = pd.read_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_decentral.csv")
		data.drop(["Unnamed: 0"], axis=1, inplace=True)
	except:
		data = pd.DataFrame(columns=["Method", "reg_coeff", "Loss1", "RMSE1", "PCC1", "Loss2", "RMSE2", "PCC2"])
	
	data = pd.concat(
		[data, pd.Series([f"{strategy_name}", f"{coeff}", truncate_float(history.losses_distributed[int(rounds / 2) - 1][-1],4),
							truncate_float(history.metrics_distributed['avg_rmse'][int(rounds / 2) - 1][-1],4),
							truncate_float(history.metrics_distributed['avg_pearson_score'][int(rounds / 2) - 1][-1],4),
							truncate_float(history.losses_distributed[-1][-1],4),
							truncate_float(history.metrics_distributed['avg_rmse'][-1][-1],4),
							truncate_float(history.metrics_distributed['avg_pearson_score'][-1][-1],4)], index=data.columns).to_frame().T])
	
	data.to_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_decentral.csv")

	try:
		try:
			data = pd.read_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_central.csv")
			data.drop(["Unnamed: 0"], axis=1, inplace=True)
		except:
			data = pd.DataFrame(columns=["Method", "reg_coeff", "Loss1", "RMSE1", "PCC1", "Loss2", "RMSE2", "PCC2"])
		
		data = pd.concat(
			[data, pd.Series([f"{strategy_name}", f"{coeff}", truncate_float(history.losses_centralized[int(rounds / 2)][-1],4),
								truncate_float(history.metrics_centralized['avg_rmse'][int(rounds / 2)][-1],4),
								truncate_float(history.metrics_centralized['avg_pearson_score'][int(rounds / 2)][-1],4),
								truncate_float(history.losses_centralized[-1][-1],4),
								truncate_float(history.metrics_centralized['avg_rmse'][-1][-1],4),
								truncate_float(history.metrics_centralized['avg_pearson_score'][-1][-1],4)],
							 index=data.columns).to_frame().T])
		data.to_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_central.csv")

	except:
		print("Centralised results not available")
	if '-' in strategy_name:
		strategy_name, base_name = strategy_name.split('-')
		if 'LGR' in base_name:
			save_path = f"{output}/{strategy_name}/{strategy_name}-{base_name}_{aug}_{clients}_{rounds}_{epochs}"
		else:
			save_path = f"{output}/{coeff}/{strategy_name}-{base_name}_{aug}_{clients}_{rounds}_{epochs}"
	else:
		save_path = f"{output}/{coeff}/{strategy_name}_{aug}_{clients}_{rounds}_{epochs}"
	
	plot_results(history=history, save_path=save_path)
	del history
	ray.shutdown()


def savecomp(output, strat, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf):
	try:
		data = pd.read_csv(f"{output}/comp.csv")
		data.drop(["Unnamed: 0"], axis=1, inplace=True)
	except:
		data = pd.DataFrame(columns=["Strategy", "RAM Before", "RAM After", "CPU Before", "CPU After", "GPU Before", "GPU After"])
	data = pd.concat([data, pd.Series([f"{strat}", rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf], index=data.columns).to_frame().T])
	data.to_csv(f"{output}/comp.csv")


def run(args):
	def client_fn(cid) -> FlowerClientCL:
		
		return FlowerClientCL(cid, net.to(DEVICE), trainloader=trainloaders, valloader=valloaders, testloader=testloader, epochs=int(args.epochs),
							  y_labels=y_labels, cl_strategy=caller, agent_config=agent_config, nrounds=int(args.rounds), path=f"{output_reg}",
							  DEVICE=DEVICE, num_clients=n_cl, strat_name=strat_cl, params=params)
	
	def client_fn_NR(cid) -> FlowerClient_NR:
		return FlowerClient_NR(cid, net.to(DEVICE), trainloader=trainloaders, valloader=valloaders, testloader=testloader, epochs=int(args.epochs),
							   y_labels=y_labels, cl_strategy=caller, agent_config=agent_config, nrounds=int(args.rounds), path=f"{output_NR}",
							   DEVICE=DEVICE, num_clients=n_cl, strat_name=strat_cl, params=params)
	
	from models.default import Net
	
	if args.reg_coef == 'all':
		reg_coefficients = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
	else:
		reg_coefficients = [float(args.reg_coef)]
	experiment_path = f"{args.output}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.strategy_fl}_{args.strategy_cl}_{args.model}_{args.rounds}_{args.icl}_{args.fcl}_{args.aug}_{args.base}_{args.processor_type}"
	if not os.path.exists(experiment_path):
		os.makedirs(experiment_path)
	# update the output path
	args.output = experiment_path
	num_CPUs = 4
	if args.processor_type == 'gpu':
		num_GPUs = 1
		ray_init_args = {"num_gpus": num_GPUs, "num_cpus": num_CPUs}
		DEVICE = torch.device("cuda")
		gpu_flag = 1
	else:
		num_GPUs = 0
		ray_init_args = {"num_gpus": num_GPUs, "num_cpus": num_CPUs}
		DEVICE = torch.device("cpu")
		gpu_flag = 0
	
	# Initial RAM and CPU Usage.
	ramu = RAMU()
	cpuu = CPUUsage()
	
	# Initial GPU usage for GPUID
	if gpu_flag == 1:
		gpuu = GPUUsage(0)
	
	# Running the loop for clients between icl (low) and fcl (high)
	for n_cl in range(int(args.icl), int(args.fcl) + 1):
		net = Net()
		if gpu_flag == 1:
			client_res = {"num_gpus": num_GPUs / n_cl, "num_cpus": num_CPUs}
		else:
			client_res = {"num_gpus": num_GPUs, "num_cpus": num_CPUs}
		
		if not os.path.exists(f"{args.output}/{n_cl}"):
			os.mkdir(f"{args.output}/{n_cl}")
		
		trainloaders, valloaders, testloader, y_labels = splitter(args.path, n_cl, args.aug)
		
		if args.strategy_fl == 'all':
			strategies = ['FedAvg']
		else:
			strategies = [args.strategy_fl]
		if args.strategy_cl == 'all':
			strategies_cl = ['EWC', 'EWCOnline', 'SI', 'MAS', 'NR']
		else:
			strategies_cl = [args.strategy_cl]
		for strat in strategies:
			params = get_parameters(net)
			path = f"{args.output}/{n_cl}/{strat}"
			if not os.path.exists(path):
				os.mkdir(path)
			if strat == 'FedAvg':
				for strat_cl in strategies_cl:
					central_model = net
					output = f"{args.output}/{n_cl}/{strat}/{strat_cl}"
					if not os.path.exists(output):
						os.mkdir(output)
					if strat_cl in ['EWC', 'EWCOnline', 'SI', 'MAS']:
						for coeff in reg_coefficients:
							output_reg = f"{output}/{coeff}"
							if not os.path.exists(f"{output_reg}"):
								os.mkdir(f"{output_reg}")
							
							agent_config = {'lr': 0.01, 'momentum': 0.1, 'weight_decay': 0.01, 'schedule': [int(args.epochs)],
											'model_type': 'mode', 'model_name': 'model', 'model_weights': '', 'out_dim': {'All': 8},
											'optimizer': 'Adam', 'print_freq': 0, 'gpuid': [gpu_flag], 'reg_coef': coeff}
							
							if strat_cl == 'EWC':
								caller = EWC
							elif strat_cl == 'EWCOnline':
								caller = EWCOnline
							elif strat_cl == 'SI':
								caller = SI
							elif strat_cl == 'MAS':
								caller = MAS
							
							strategy = FedAvgWithAccuracyMetric(
								min_available_clients=int(n_cl),
								initial_parameters=fl.common.ndarrays_to_parameters(params),
								on_fit_config_fn=fit_config,
								on_evaluate_config_fn=evaluate_config,
								evaluate_fn=get_eval_fn(central_model, testloader=valloaders, DEVICE=DEVICE, y_labels=y_labels)
							
							)
							rambef = ramu.compute("BEFORE EVALUATION")
							cpubef = cpuu.compute("BEFORE EVALUATION")
							if gpu_flag == 1:
								gpubeff = gpuu.compute("BEFORE EVALUATION")
							else:
								gpubeff = 0
							run_strategy(strategy, f"{strat_cl}", coeff, client_fn, n_cl, int(args.rounds), int(args.epochs),
										 f"{output}", args.aug, ray_init_args, client_res)
							ramaf = ramu.compute("AFTER EVALUATION")
							cpuaf = cpuu.compute("AFTER EVALUATION")
							if gpu_flag == 1:
								gpuaf = gpuu.compute("AFTER EVALUATION")
							else:
								gpuaf = 0
							savecomp(f"{output}", coeff, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
					elif strat_cl == 'NR':
						if args.reg_coef == 'all':
							buffer_sizes = [1, 10, 100, 1000, 10000]
						else:
							buffer_sizes = [int(args.reg_coef)]
							if buffer_sizes[0] == 0:
								buffer_sizes = [1]
						for buffer_size in buffer_sizes:
							output_NR = f"{output}/{buffer_size}"
							if not os.path.exists(f"{output_NR}"):
								os.mkdir(f"{output_NR}")
							
							agent_config = {'lr': 0.01, 'momentum': 0.1, 'weight_decay': 0.01, 'schedule': [int(args.epochs)],
											'model_type': 'mode', 'model_name': 'model', 'model_weights': '', 'out_dim': {'All': 8},
											'optimizer': 'Adam', 'print_freq': 0, 'gpuid': [gpu_flag], 'memory_size': buffer_size, 'reg_coef': 0.01}
							caller = NR
							
							strategy = FedAvgWithAccuracyMetric(
								min_available_clients=int(n_cl),
								initial_parameters=fl.common.ndarrays_to_parameters(params),
								on_fit_config_fn=fit_config,
								on_evaluate_config_fn=evaluate_config,
								evaluate_fn=get_eval_fn(central_model, testloader=valloaders, DEVICE=DEVICE, y_labels=y_labels)
							
							)
							rambef = ramu.compute("BEFORE EVALUATION")
							cpubef = cpuu.compute("BEFORE EVALUATION")
							if gpu_flag == 1:
								gpubeff = gpuu.compute("BEFORE EVALUATION")
							else:
								gpubeff = 0
							run_strategy(strategy, f"{strat_cl}", buffer_size, client_fn_NR, n_cl, int(args.rounds), int(args.epochs),
										 f"{output}", args.aug, ray_init_args, client_res)
							ramaf = ramu.compute("AFTER EVALUATION")
							cpuaf = cpuu.compute("AFTER EVALUATION")
							if gpu_flag == 1:
								gpuaf = gpuu.compute("AFTER EVALUATION")
							else:
								gpuaf = 0
							savecomp(f"{output}", buffer_size, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
					df2 = extract_metrics_gpu_csv(f"{output}/comp.csv")
					df2.to_csv(f"{output}/comp_extracted.csv")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Argument Parser for FCL.")
	parser.add_argument("-sfl", "--strategy_fl", type=str, default='all', help="FL Strategy to use")
	parser.add_argument("-scl", "--strategy_cl", type=str, default='all', help="CL Strategy to use")
	parser.add_argument("-m", "--model", type=str, default='MobileNet', help="Model to use")
	parser.add_argument("-r", "--reg_coef", type=str, default='all', help="Regularisation Coefficient.")
	parser.add_argument("-n", "--rounds", type=int, default=10, help="Number of Rounds")
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of Epochs")
	parser.add_argument("-x", "--batch_size", default=16, type=int, help="Batch Size?")
	parser.add_argument("-c", "--icl", type=int, default=2, help="Initial number of clients")
	parser.add_argument("-f", "--fcl", type=int, default=10, help="Final number of clients")
	parser.add_argument("-p", "--path", type=str, help="Path to dataset")
	parser.add_argument("-o", "--output", type=str, help="Output path")
	parser.add_argument("-a", "--aug", type=eval, choices=[True, False], default='False', help="Use Augmentation?")
	parser.add_argument("-b", "--base", type=str, default="FedAvg", help="Default base for FedRoot Only")
	parser.add_argument("-t", "--processor_type", type=str, default="cpu", help="Processor Type")
	args = parser.parse_args()
	
	print("Running with the following arguments:")
	print("Strategy FL: ", args.strategy_fl)
	print("Strategy CL: ", args.strategy_cl)
	print("Model: ", args.model)
	print("Rounds: ", args.rounds)
	print("Batch_size: ", args.batch_size)
	print("Clients start: ", args.icl)
	print("Clients end: ", args.fcl)
	print("Path: ", args.path)
	print("Output: ", args.output)
	print("Augmentation: ", args.aug)
	print("Base: ", args.base)
	print("Processor Type: ", args.processor_type)
	
	run(args)

# python3 main.py -sfl FedAvg -scl EWC -m default -r 0.01 -n 10 -e 10 -x 16 -c 2 -f 2 -p SADRA-Dataset -o output -a False -b FedAvg -t cpu
