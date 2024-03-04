from __future__ import print_function
import torch
import torch.nn as nn
import random
from .utils import AverageMeter, Timer, accuracy
import sys
import torch.utils.data as data
import torch.nn.functional as F
from scipy.stats import pearsonr
import math
from utils import get_parameters, pearson_correlation, train, test, predict, predict_gen, RMSELoss
from torch.utils.data import ConcatDataset, DataLoader
import warnings
from collections import OrderedDict
import numpy as np

warnings.filterwarnings("ignore")

sys.path.append('..')
from dataloader.outloader import CustomOutputDataset


def Average(lst):
	return sum(lst) / len(lst)


class NormalNN(nn.Module):
	'''
	Normal Neural Network with SGD for classification
	'''
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		'''
		:param agent_config (dict): lr=float,momentum=float,weight_decay=float,
									schedule=[int],  # The last number in the list is the end of epoch
									model_type=str,model_name=str,out_dim={task:dim},model_weights=str
									force_single_head=bool
									print_freq=int
									gpuid=[int]
		'''
		super(NormalNN, self).__init__()
		# self.Net = Net
		self.log = print if agent_config['print_freq'] > 0 else lambda \
				*args: None  # Use a void function to replace the print
		self.config = agent_config
		if agent_config['gpuid'][0] > 0:
			self.gpu = True
		else:
			self.gpu = False
		# If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
		self.multihead = True if len(self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
		self.model = self.create_model(model=net, params=params, fedroot=fedroot)
		self.criterion_fn = nn.MSELoss()
		# self.criterion_fn = nn.L1Loss()
		if self.gpu:
			self.cuda()
		
		self.init_optimizer()
		self.reset_optimizer = False
		self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
		self.criterion_rmse = RMSELoss()
	
	# Set a interger here for the incremental class scenario
	
	def init_optimizer(self):
		optimizer_arg = {'params': self.model.parameters(),
		                 'lr': self.config['lr'],
		                 'weight_decay': self.config['weight_decay']}
		if self.config['optimizer'] in ['SGD', 'RMSprop']:
			optimizer_arg['momentum'] = self.config['momentum']
		elif self.config['optimizer'] in ['Rprop']:
			optimizer_arg.pop('weight_decay')
		elif self.config['optimizer'] == 'amsgrad':
			optimizer_arg['amsgrad'] = True
			self.config['optimizer'] = 'Adam'
		
		self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'], gamma=0.1)
	
	def create_model(self, model, params=None, fedroot=False):
		if self.gpu:
			model.cuda()
		if params is not None:
			if not fedroot:
				params_dict = zip(model.state_dict().keys(), params)
				state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
				model.load_state_dict(state_dict)
			else:
				params_dict = zip(model.conv_module.state_dict().keys(), params)
				state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
				model.conv_module.load_state_dict(state_dict, strict=True)
		return model
	
	def forward(self, x):
		return self.model.forward(x)
	
	def predict(self, inputs):
		self.model.eval()
		out = self.forward(inputs)
		for t in out.keys():
			out[t] = out[t].detach()
		return out
	
	def validation(self, dataloader):
		# This function doesn't distinguish tasks.
		batch_timer = Timer()
		acc = AverageMeter()
		batch_timer.tic()
		
		orig_mode = self.training
		self.eval()
		for i, (inputs, target, task) in enumerate(dataloader):
			
			if self.gpu:
				with torch.no_grad():
					inputs = inputs.cuda()
					target = target.cuda()
			output = self.predict(inputs)
			
			# Summarize the performance of all tasks, or 1 task, depends on dataloader.
			# Calculated by total number of data.
			acc = accumulate_acc(output, target, task, acc)
		
		self.train(orig_mode)
		
		self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
		         .format(acc=acc, time=batch_timer.toc()))
		return acc.avg
	
	def criterion(self, preds, targets, **kwargs):
		loss = self.criterion_fn(preds, targets)
		return loss
	
	def update_model(self, inputs, targets):
		self.optimizer.zero_grad()
		out = self.forward(inputs)
		loss = self.criterion(out, targets)
		loss.backward()
		self.optimizer.step()
		return loss.detach(), out
	
	def learn_batch(self, train_loader, val_loader=None):
		if self.reset_optimizer:  # Reset optimizer before learning each task
			self.log('Optimizer is reset!')
			self.init_optimizer()
		
		y_labels = [0, 1, 2, 3, 4, 5, 6, 7]
		
		for epoch in range(self.config['schedule'][-1]):
			data_timer = Timer()
			batch_timer = Timer()
			batch_time = AverageMeter()
			data_time = AverageMeter()
			losses = AverageMeter()
			# pcc = AverageMeter()
			
			# Config the model and optimizer
			self.log('Epoch:{0}'.format(epoch))
			self.model.train()
			self.scheduler.step(epoch)
			for param_group in self.optimizer.param_groups:
				self.log('LR:', param_group['lr'])
			
			# Learning with mini-batch
			data_timer.tic()
			batch_timer.tic()
			self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
			for i, (inputs, targets) in enumerate(train_loader):
				data_time.update(data_timer.toc())  # measure data loading time
				
				if self.gpu:
					inputs = inputs.cuda()
					targets = targets.cuda()
				
				loss, output = self.update_model(inputs, targets)
				inputs_size = inputs.detach().cpu().numpy().size
				targets = targets.detach().cpu().numpy()
				output = output.detach().cpu().numpy()
				# calculate pearson correlation
				# pcc.update(pearson_correlation(target, output), input.size(0))
				
				losses.update(loss, inputs_size)
				
				batch_time.update(batch_timer.toc())  # measure elapsed time
				data_timer.toc()
			print(f"Epoch {epoch + 1} / {self.config['schedule'][-1]} = {losses.avg}")
			# print(f"Loss : Epoch {epoch + 1} / {self.config['schedule'][-1]} = {losses.avg}")
			# print(f"PCC: Epoch {epoch + 1} / {self.config['schedule'][-1]} = {Average(list(pearson.values()))}")
			# Evaluate the performance of current task
			if val_loader is not None:
				self.validation(val_loader)
	
	def test(self, test_loader):
		# loss, rmse, pearson
		losses = AverageMeter()
		pcc = AverageMeter()
		rmse = AverageMeter()
		self.model.eval()
		for i, (inputs, target) in enumerate(test_loader):
			if self.gpu:
				inputs = inputs.cuda()
				target = target.cuda()
			output = self.model(inputs)
			losses.update(self.criterion(output, target), inputs.size(0))
			inputs = inputs.detach()
			target = target.detach()
			output = output.detach()
			if self.gpu:
				inputs = inputs.cpu()
				target = target.cpu()
				output = output.cpu()
			pcc.update(pearson_correlation(target, output), inputs.size(0))
			rmse.update(self.criterion_rmse(target, output), inputs.size(0))
		return losses.avg, pcc.avg, rmse.avg
	
	def learn_stream(self, data, label):
		assert False, 'No implementation yet'
	
	def add_valid_output_dim(self, dim=0):
		# This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
		self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
		if self.valid_out_dim == 'ALL':
			self.valid_out_dim = 0  # Initialize it with zero
		self.valid_out_dim += dim
		self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
		return self.valid_out_dim
	
	def count_parameter(self):
		return sum(p.numel() for p in self.model.parameters())
	
	def save_model(self, filename):
		model_state = self.model.state_dict()
		if isinstance(self.model, torch.nn.DataParallel):
			# Get rid of 'module' before the name of states
			model_state = self.model.module.state_dict()
		for key in model_state.keys():  # Always save it to cpu
			model_state[key] = model_state[key].cpu()
		print('=> Saving model to:', filename)
		torch.save(model_state, filename + '.pth')
		print('=> Save Done')
	
	def cuda(self):
		# torch.cuda.set_device(self.config['gpuid'][0])
		self.model = self.model.cuda()
		self.criterion_fn = self.criterion_fn.cuda()
		# Multi-GPU
		if len(self.config['gpuid']) > 1:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
		return self
	
	def load_model(self, statedict):
		self.model.train()
		params_dict = zip(self.model.state_dict().keys(), statedict)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict)
	
	def get(self):
		return get_parameters(self.model)


def accumulate_acc(output, target, task, meter):
	if 'All' in output.keys():  # Single-headed model
		meter.update(accuracy(output['All'], target), len(target))
	else:  # outputs from multi-headed (multi-task) model
		for t, t_out in output.items():
			inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
			if len(inds) > 0:
				t_out = t_out[inds]
				t_target = target[inds]
				meter.update(accuracy(t_out, t_target), len(inds))
	
	return meter


class L2(NormalNN):
	"""
	@article{kirkpatrick2017overcoming,
		title={Overcoming catastrophic forgetting in neural networks},
		author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
		journal={Proceedings of the national academy of sciences},
		year={2017},
		url={https://arxiv.org/abs/1612.00796}
	}
	"""
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		super(L2, self).__init__(agent_config, net, params, fedroot)
		self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
		self.regularization_terms = {}
		self.task_count = 0
		self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
	
	# False: Each task has its own importance matrix and model parameters
	
	def calculate_importance(self, dataloader):
		# Use an identity importance so it is an L2 regularization.
		importance = {}
		for n, p in self.params.items():
			importance[n] = p.clone().detach().fill_(1)  # Identity
		return importance
	
	def learn_batch(self, task_count, regularization_terms, train_loader, learn_ewc, val_loader=None):
		# self.model=net
		
		self.task_count = task_count
		self.regularization_terms = regularization_terms
		self.log('#reg_term:', len(self.regularization_terms))
		
		# 1.Learn the parameters for current task
		super(L2, self).learn_batch(train_loader, val_loader)
		
		# 2.Backup the weight of current task
		task_param = {}
		for n, p in self.params.items():
			task_param[n] = p.clone().detach()
		
		if learn_ewc:
			# 3.Calculate the importance of weights for current task
			importance = self.calculate_importance(train_loader)
			
			# Save the weight and importance of weights of current task
			# self.task_count += 1
			if self.online_reg and len(self.regularization_terms) > 0:
				# Always use only one slot in self.regularization_terms
				self.regularization_terms[1] = {'importance': importance, 'task_param': task_param}
			else:
				# Use a new slot to store the task-specific information
				self.regularization_terms[self.task_count] = {'importance': importance, 'task_param': task_param}
		return self.model
	
	def criterion(self, preds, targets, regularization=True, **kwargs):
		loss = super(L2, self).criterion(preds, targets, **kwargs)
		if self.task_count > 0:
			if regularization and len(self.regularization_terms) > 0:
				self.log("Running Criterion using Importance")
				# Calculate the reg_loss only when the regularization_terms exists
				reg_loss = 0
				for i, reg_term in self.regularization_terms.items():
					task_reg_loss = 0
					importance = reg_term['importance']
					task_param = reg_term['task_param']
					for n, p in self.params.items():
						task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
					reg_loss += task_reg_loss
				loss += self.config['reg_coef'] * reg_loss
		return loss


class EWC(L2):
	"""
	@article{kirkpatrick2017overcoming,
		title={Overcoming catastrophic forgetting in neural networks},
		author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
		journal={Proceedings of the national academy of sciences},
		year={2017},
		url={https://arxiv.org/abs/1612.00796}
	}
	"""
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		super(EWC, self).__init__(agent_config, net, params, fedroot)
		self.online_reg = False
		self.n_fisher_sample = None
		self.empFI = False
	
	def calculate_importance(self, dataloader):
		# Update the diag fisher information
		# There are several ways to estimate the F matrix.
		# We keep the implementation as simple as possible while maintaining a similar performance to the literature.
		self.log('Computing EWC')
		print('Computing EWC Importance')
		self.n_fisher_sample = int(len(dataloader) * 0.67)
		# Initialize the importance matrix
		if self.online_reg and len(self.regularization_terms) > 0:
			importance = self.regularization_terms[1]['importance']
		else:
			importance = {}
			for n, p in self.params.items():
				importance[n] = p.clone().detach().fill_(0)  # zero initialized
		
		# Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
		# Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
		if self.n_fisher_sample is not None:
			n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
			self.log('Sample', self.n_fisher_sample, 'for estimating the F matrix.')
			rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
			subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
			dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=2, batch_size=1)
		
		mode = self.training
		self.eval()
		
		# Accumulate the square of gradients
		for i, (inputs, target) in enumerate(dataloader):
			if self.gpu:
				inputs = inputs.cuda()
				target = target.cuda()
			self.model.zero_grad()
			preds = self.forward(inputs)
			loss = self.criterion(preds, target, regularization=False)
			loss.backward()
			for n, p in importance.items():
				if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
					p += ((self.params[n].grad ** 2) * len(inputs) / len(dataloader))
		
		self.train(mode=mode)
		
		return importance


class EWCOnline(L2):
	"""
	@article{kirkpatrick2017overcoming,
		title={Overcoming catastrophic forgetting in neural networks},
		author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
		journal={Proceedings of the national academy of sciences},
		year={2017},
		url={https://arxiv.org/abs/1612.00796}
	}
	"""
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		super(EWCOnline, self).__init__(agent_config, net, params, fedroot)
		self.online_reg = True
		self.n_fisher_sample = None
		self.empFI = False
	
	def calculate_importance(self, dataloader):
		# Update the diag fisher information
		# There are several ways to estimate the F matrix.
		# We keep the implementation as simple as possible while maintaining a similar performance to the literature.
		self.log('Computing EWC')
		
		# Initialize the importance matrix
		if self.online_reg and len(self.regularization_terms) > 0:
			importance = self.regularization_terms[1]['importance']
		else:
			importance = {}
			for n, p in self.params.items():
				importance[n] = p.clone().detach().fill_(0)  # zero initialized
		
		# Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
		# Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
		if self.n_fisher_sample is not None:
			n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
			self.log('Sample', self.n_fisher_sample, 'for estimating the F matrix.')
			rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
			subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
			dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=2, batch_size=1)
		
		mode = self.training
		self.eval()
		
		# Accumulate the square of gradients
		for i, (inputs, target) in enumerate(dataloader):
			if self.gpu:
				inputs = inputs.cuda()
				target = target.cuda()
			
			preds = self.forward(inputs)
			
			loss = self.criterion(preds, target, regularization=False)
			self.model.zero_grad()
			loss.backward()
			for n, p in importance.items():
				if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
					p += ((self.params[n].grad ** 2) * len(inputs) / len(dataloader))
		
		self.train(mode=mode)
		
		return importance


class Storage(data.Dataset):
	"""
	A dataset wrapper used as a memory to store the data
	"""
	
	def __init__(self):
		super(Storage, self).__init__()
		self.storage = []
	
	def __len__(self):
		return len(self.storage)
	
	def __getitem__(self, index):
		return self.storage[index]
	
	def append(self, x):
		self.storage.append(x)
	
	def extend(self, x):
		self.storage.extend(x)
	
	def update(self, li):
		self.storage = li


class Memory(Storage):
	def reduce(self, m):
		self.storage = self.storage[:m]


# class Naive_Rehearsal(NormalNN):
#
# 	def __init__(self, agent_config, Net):
# 		super(Naive_Rehearsal, self).__init__(agent_config, Net)
# 		self.task_count = 0
# 		self.task_memory = {}
# 		self.memory_size = self.config['memory_size']
#
# 	def learn_batch(self, task_count, memory, train_loader, learn_nr, val_loader=None):
# 		# 1.Combine training set
# 		self.task_count = task_count
# 		self.task_memory = memory
# 		if self.task_count != 0:
# 			dataset_list = []
# 			for storage in self.task_memory.values():
# 				dataset_list.append(storage)
# 			dataset_list *= max(len(train_loader.dataset) // self.memory_size, 1)  # Let old data: new data = 1:1
# 			dataset_list.append(train_loader.dataset)
# 			dataset = torch.utils.data.ConcatDataset(dataset_list)
# 			new_train_loader = torch.utils.data.DataLoader(dataset,
# 			                                               batch_size=train_loader.batch_size,
# 			                                               shuffle=True,
# 			                                               num_workers=train_loader.num_workers)
# 			# 2.Update model as normal
# 			super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader)
# 		else:
# 			super(Naive_Rehearsal, self).learn_batch(train_loader, val_loader)
#
# 		# 3.Randomly decide the images to stay in the memory
# 		if learn_nr:
# 			self.update_memory(train_loader)
#
# 		return self.model
#
# 	def update_memory(self, train_loader):
# 		# (a) Decide the number of samples for being saved
# 		num_sample_per_task = self.memory_size // self.task_count
# 		num_sample_per_task = min(len(train_loader.dataset), num_sample_per_task)
# 		# (b) Reduce current exemplar set to reserve the space for the new dataset
# 		for storage in self.task_memory.values():
# 			storage.reduce(num_sample_per_task)
# 		# (c) Randomly choose some samples from new task and save them to the memory
# 		self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
# 		randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
# 		for ind in randind:  # save it to the memory
# 			self.task_memory[self.task_count].append(train_loader.dataset[ind])


class SI(L2):
	"""
	@inproceedings{zenke2017continual,
		title={Continual Learning Through Synaptic Intelligence},
		author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
		booktitle={International Conference on Machine Learning},
		year={2017},
		url={https://arxiv.org/abs/1703.04200}
	}
	"""
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		super(SI, self).__init__(agent_config, net, params, fedroot)
		self.online_reg = True  # Original SI works in an online updating fashion
		self.damping_factor = 0.1
		self.w = {}
		for n, p in self.params.items():
			self.w[n] = p.clone().detach().zero_()
		
		# The initial_params will only be used in the first task (when the regularization_terms is empty)
		self.initial_params = {}
		for n, p in self.params.items():
			self.initial_params[n] = p.clone().detach()
	
	def update_model(self, inputs, targets):
		
		unreg_gradients = {}
		
		# 1.Save current parameters
		old_params = {}
		for n, p in self.params.items():
			old_params[n] = p.clone().detach()
		
		# 2. Collect the gradients without regularization term
		out = self.forward(inputs)
		loss = self.criterion(out, targets, regularization=False)
		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		for n, p in self.params.items():
			if p.grad is not None:
				unreg_gradients[n] = p.grad.clone().detach()
		
		# 3. Normal update with regularization
		loss = self.criterion(out, targets, regularization=True)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		# 4. Accumulate the w
		for n, p in self.params.items():
			delta = p.detach() - old_params[n]
			if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
				self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0
		
		return loss.detach(), out
	
	"""
	# - Alternative simplified implementation with similar performance -
	def update_model(self, inputs, targets, tasks):
		# A wrapper of original update step to include the estimation of w

		# Backup prev param if not done yet
		# The backup only happened at the beginning of a new task
		if len(self.prev_params) == 0:
			for n, p in self.params.items():
				self.prev_params[n] = p.clone().detach()

		# 1.Save current parameters
		old_params = {}
		for n, p in self.params.items():
			old_params[n] = p.clone().detach()

		# 2.Calculate the loss as usual
		loss, out = super(SI, self).update_model(inputs, targets, tasks)

		# 3.Accumulate the w
		for n, p in self.params.items():
			delta = p.detach() - old_params[n]
			if p.grad is not None:  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
				self.w[n] -= p.grad * delta  # w[n] is >=0

		return loss.detach(), out
	"""
	
	def calculate_importance(self, dataloader):
		self.log('Computing SI')
		print('Computing SI')
		assert self.online_reg, 'SI needs online_reg=True'
		
		# Initialize the importance matrix
		if len(self.regularization_terms) > 0:  # The case of after the first task
			importance = self.regularization_terms[1]['importance']
			prev_params = self.regularization_terms[1]['task_param']
		else:  # It is in the first task
			importance = {}
			for n, p in self.params.items():
				importance[n] = p.clone().detach().fill_(0)  # zero initialized
			prev_params = self.initial_params
		
		# Calculate or accumulate the Omega (the importance matrix)
		for n, p in importance.items():
			delta_theta = self.params[n].detach() - prev_params[n]
			p += self.w[n] / (delta_theta ** 2 + self.damping_factor)
			self.w[n].zero_()
		
		return importance


class MAS(L2):
	"""
	@article{aljundi2017memory,
	  title={Memory Aware Synapses: Learning what (not) to forget},
	  author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
	  booktitle={ECCV},
	  year={2018},
	  url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
	}
	"""
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		super(MAS, self).__init__(agent_config, net, params, fedroot)
		self.online_reg = True
	
	def calculate_importance(self, dataloader):
		self.log('Computing MAS')
		
		# Initialize the importance matrix
		if self.online_reg and len(self.regularization_terms) > 0:
			importance = self.regularization_terms[1]['importance']
		else:
			importance = {}
			for n, p in self.params.items():
				importance[n] = p.clone().detach().fill_(0)  # zero initialized
		
		mode = self.training
		self.eval()
		
		# Accumulate the gradients of L2 loss on the outputs
		for i, (inputs, target) in enumerate(dataloader):
			if self.gpu:
				inputs = inputs.cuda()
				target = target.cuda()
			
			preds = self.forward(inputs)
			
			preds.pow_(2)
			loss = preds.mean()
			
			self.model.zero_grad()
			loss.backward()
			for n, p in importance.items():
				if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
					p += (self.params[n].grad.abs() / len(dataloader))
		
		self.train(mode=mode)
		
		return importance


class Naive_Rehearsal(NormalNN):
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		super(Naive_Rehearsal, self).__init__(agent_config, net, params, fedroot)
		self.task_count = 0
		self.task_memory = {}
		self.memory_size = self.config['memory_size']
	
	def learn_batch(self, task_count, memory, train_loader, learn_nr, val_loader=None):
		# 1.Combine training set
		self.task_count = task_count
		self.task_memory = memory
		if self.task_memory != {} and self.task_count > 0:
			dataset_list = []
			for storage in self.task_memory.values():
				dataset_list.append(storage)
			dataset_list *= max(len(train_loader.dataset) // self.memory_size, 1)  # Let old data: new data = 1:1
			dataset_list.append(train_loader.dataset)
			dataset = torch.utils.data.ConcatDataset(dataset_list)
			new_train_loader = torch.utils.data.DataLoader(dataset,
			                                               batch_size=train_loader.batch_size,
			                                               shuffle=True,
			                                               num_workers=train_loader.num_workers,
														   drop_last=True)
			# 2.Update model as normal
			super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader)
		else:
			super(Naive_Rehearsal, self).learn_batch(train_loader, val_loader)
		
		# 3.Randomly decide the images to stay in the memory
		if learn_nr:
			self.update_memory(train_loader)
		
		return self.model
	
	def update_memory(self, train_loader):
		# (a) Decide the number of samples for being saved
		num_sample_per_task = self.memory_size // self.task_count
		num_sample_per_task = min(len(train_loader.dataset), num_sample_per_task)
		# (b) Reduce current exemplar set to reserve the space for the new dataset
		for storage in self.task_memory.values():
			storage.reduce(num_sample_per_task)
		# (c) Randomly choose some samples from new task and save them to the memory
		self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
		randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
		randind = randind.detach().cpu().numpy()
		for ind in randind:  # save it to the memory
			self.task_memory[self.task_count].append(train_loader.dataset[ind])


class LatentGenerativeReplay(nn.Module):
	def __init__(self, agent_config, net, params, Gen, path, client_id):
		super(LatentGenerativeReplay, self).__init__()
		# self.Net = Net
		self.generator = Gen
		self.log = print if agent_config['print_freq'] > 0 else lambda *args: None  # Use a void function to replace the print
		self.config = agent_config
		if agent_config['gpuid'][0] > 0:
			
			self.gpu = True
			self.Device = torch.device("cuda")
		else:
			self.gpu = False
			self.Device = torch.device("cpu")
		# If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
		self.multihead = True if len(self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
		self.model = self.create_model(model=net, params=params)
		# self.generator=self,create_generator()
		self.criterion_fn = nn.MSELoss()
		# self.criterion_fn = nn.L1Loss()
		self.init_optimizer()
		if self.gpu:
			self.cuda()
		self.reset_optimizer = False
		self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
		self.criterion = nn.MSELoss()
		# self.criterion = nn.L1Loss()
		self.path = path
		self.client_id = client_id
	
	# Set a interger here for the incremental class scenario
	def get_generator_weights(self):
		return self.generator.state_dict()
	
	def update_model(self, inputs, targets):
		# self.model=copy.deepcopy(model_)
		out = self.forward(inputs)
		loss = self.criterion(out, targets)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.detach(), out
	
	def init_optimizer(self):
		optimizer_arg = {'params': self.model.parameters(),
		                 'lr': self.config['lr'],
		                 'weight_decay': self.config['weight_decay']}
		if self.config['optimizer'] in ['SGD', 'RMSprop']:
			optimizer_arg['momentum'] = self.config['momentum']
		elif self.config['optimizer'] in ['Rprop']:
			optimizer_arg.pop('weight_decay')
		elif self.config['optimizer'] == 'amsgrad':
			optimizer_arg['amsgrad'] = True
			self.config['optimizer'] = 'Adam'
		
		self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'], gamma=0.1)
	
	def create_model(self, model, params=None):
		if self.gpu:
			model.cuda()
		if params is not None:
			params_dict = zip(model.conv_module.state_dict().keys(), params)
			state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
			model.conv_module.load_state_dict(state_dict, strict=True)
		return model
	
	def forward(self, x):
		return self.model.forward(x)
	
	def set_generator_weights(self, weights):
		self.generator.load_state_dict(weights)
	
	def predict_gen(self, net, trainloader, DEVICE, batch_size=16):
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
	
	def predict_from_gen(self, net, num_samples, DEVICE, batch_size=16):
		new_pairs = []
		net.eval()  # Set the model to evaluation mode
		net.to(DEVICE)
		with torch.no_grad():
			for i in range(num_samples):
				inputs = torch.randn(1, 64).to(DEVICE)
				outputs = net.decode(inputs)
				labels = self.model.fc_module(outputs)
				for i in range(len(outputs)):
					new_pairs.append((outputs[i], labels[i]))
		# batch_size = 16
		new_data = CustomOutputDataset(new_pairs)
		new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
		return new_data_loader
	
	def predict_from_gen_gen(self, net, num_samples, DEVICE, batch_size=16):
		new_pairs = []
		net.eval()  # Set the model to evaluation mode
		net.to(DEVICE)
		with torch.no_grad():
			for i in range(num_samples):
				inputs = torch.randn(1, 64).to(DEVICE)
				outputs = net.decode(inputs)
				labels = self.model.fc_module(outputs)
				for i in range(len(outputs)):
					new_pairs.append((outputs[i], outputs[i]))
		# batch_size = 16
		new_data = CustomOutputDataset(new_pairs)
		new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
		return new_data_loader
	
	def create_dataset(self, new_data):
		try:
			current_task_reconstucted_data = torch.load(f'{self.path}/{self.client_id}_current_task_reconstucted_data.pth')
		except:
			current_task_reconstucted_data = self.predict_from_gen(self.generator, num_samples=len(new_data) * 16, DEVICE=self.Device, batch_size=16)
			torch.save(current_task_reconstucted_data, f'{self.path}/{self.client_id}_current_task_reconstucted_data.pth')
		# shuffle it with train_loader and return
		# both are dataloaders
		dataset1 = new_data.dataset
		dataset2 = current_task_reconstucted_data.dataset
		# for input, label in new_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		# for input, label in current_task_reconstucted_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		
		# Combine the datasets using ConcatDataset
		combined_dataset = ConcatDataset([dataset1, dataset2])
		
		# Create a DataLoader for the combined dataset with shuffling
		combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)
		return combined_dataloader
	
	def create_dataset_gen(self, new_data):
		try:
			current_task_reconstucted_data = torch.load(f'{self.path}/{self.client_id}_current_task_reconstucted_data_generator.pth')
		except:
			current_task_reconstucted_data = self.predict_from_gen_gen(self.generator, num_samples=len(new_data) * 16, DEVICE=self.Device,
			                                                           batch_size=16)
			torch.save(current_task_reconstucted_data, f'{self.path}/{self.client_id}_current_task_reconstucted_data_generator.pth')
		# shuffle it with train_loader and return
		# both are dataloaders
		dataset1 = new_data.dataset
		dataset2 = current_task_reconstucted_data.dataset
		# for input, label in new_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		# for input, label in current_task_reconstucted_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		
		# Combine the datasets using ConcatDataset
		combined_dataset = ConcatDataset([dataset1, dataset2])
		
		# Create a DataLoader for the combined dataset with shuffling
		combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)
		return combined_dataloader
	
	def loss_function(self, recon_x, x, mu, logvar, input_dim):
		# MSE reconstruction loss
		# MSE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
		MSE = F.mse_loss(input=x.view(-1, input_dim), target=recon_x, reduction='mean')
		# KL divergence
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return MSE + KLD
	
	def train_generator(self, train_loader, task_count=0):
		print(" ............................................................................ Learning LGR Training Generator")
		
		# create dataset from latent features of self.model.root
		self.generator.train()
		self.generator.to(self.Device)
		optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.01)
		gen_train_data = predict_gen(self.model.conv_module, train_loader, self.Device, batch_size=16)
		if task_count != 0:
			gen_train_data = self.create_dataset_gen(gen_train_data)
		tot = self.config['schedule'][-1]
		for epoch in range(self.config['schedule'][-1] // 2):
			for input, target in gen_train_data:
				if self.gpu:
					input = input.cuda()
					target = target.cuda()
				recon_batch, mu, logvar = self.generator.forward(input)
				loss = self.loss_function(recon_batch, input, mu, logvar, input.shape[1])
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			
			# print loss
			print(f'Epoch [{epoch + 1}/{tot}], Loss: {loss.item():.4f}')
	
	def latent_creator(self, train_loader):
		self.model.eval()
		self.model.to(self.Device)
		current_task_latent_data = predict(self.model.conv_module, train_loader, self.Device, batch_size=16)
		print(" ............................................................................ Learning LGR Data Mixed")
		
		final = self.create_dataset(current_task_latent_data)
		return final
	
	def learn_batch(self, task_count, genweights, train_loader, learn_gen, val_loaderr=None):
		self.generator.load_state_dict(genweights)
		self.task_count = task_count
		self.log("Learning LGR")
		
		if self.task_count == 0:
			print(" ............................................................................ Learning LGR Task 1")
			# Config the model and optimizer
			self.model.train()
			params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.01}]
			
			for epoch in range(self.config['schedule'][-1]):
				self.log('Epoch:{0}'.format(epoch))
				data_timer = Timer()
				batch_timer = Timer()
				batch_time = AverageMeter()
				data_time = AverageMeter()
				losses = AverageMeter()
				acc = AverageMeter()
				self.scheduler.step(epoch)
				optimizer = torch.optim.Adam(params)
				
				# Learning with mini-batch
				data_timer.tic()
				batch_timer.tic()
				self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
				for i, (inputs, targets) in enumerate(train_loader):
					data_time.update(data_timer.toc())
					if self.gpu:
						inputs = inputs.cuda()
						targets = targets.cuda()
					out = self.forward(inputs)
					loss = self.criterion(out, targets)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					
					inputs = inputs.detach()
					targets = targets.detach()
					acc = 0
					losses.update(loss, inputs.size(0))
					batch_time.update(batch_timer.toc())
					data_timer.toc()
				print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
		else:
			print(" ............................................................................ Learning LGR Task 2")
			
			# train(self.model.fc_module, train_loader, self.Device, self.config['schedule'][-1])
			mixed_task_data = self.latent_creator(train_loader)
			self.model.train()
			
			for epoch in range(self.config['schedule'][-1]):
				data_timer = Timer()
				batch_timer = Timer()
				batch_time = AverageMeter()
				data_time = AverageMeter()
				losses = AverageMeter()
				acc = AverageMeter()
				
				# Config the model and optimizer
				self.log('Epoch:{0}'.format(epoch))
				self.scheduler.step(epoch)
				optimizer = torch.optim.Adam(self.model.fc_module.parameters(), lr=0.01)
				# Learning with mini-batch
				data_timer.tic()
				batch_timer.tic()
				self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
				for i, (inputs, targets) in enumerate(mixed_task_data):
					data_time.update(data_timer.toc())
					if self.gpu:
						inputs = inputs.cuda()
						targets = targets.cuda()
					# dont use self.update_model
					# dont use self.forward
					out = self.model.fc_module(inputs)
					loss = self.criterion(out, targets)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					inputs = inputs.detach()
					targets = targets.detach()
					acc = 0
					losses.update(loss, inputs.size(0))
					batch_time.update(batch_timer.toc())
					data_timer.toc()
				print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
			# freeze all layers of self.model.fc_module
			print(" ............................................................................ Learning LGR End to End Top Frozen")
			for epoch in range(self.config['schedule'][-1]):
				data_timer = Timer()
				batch_timer = Timer()
				batch_time = AverageMeter()
				data_time = AverageMeter()
				losses = AverageMeter()
				acc = AverageMeter()
				
				# Config the model and optimizer
				self.log('Epoch:{0}'.format(epoch))
				self.model.train()
				# params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.001}]
				params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.0}]
				self.scheduler.step(epoch)
				optimizer = torch.optim.Adam(params)
				
				# Learning with mini-batch
				data_timer.tic()
				batch_timer.tic()
				self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
				for i, (inputs, targets) in enumerate(train_loader):
					data_time.update(data_timer.toc())
					if self.gpu:
						inputs = inputs.cuda()
						targets = targets.cuda()
					out = self.forward(inputs)
					loss = self.criterion(out, targets)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					
					inputs = inputs.detach()
					targets = targets.detach()
					acc = 0
					losses.update(loss, inputs.size(0))
					batch_time.update(batch_timer.toc())
					data_timer.toc()
				print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
		
		if learn_gen == True:
			self.train_generator(train_loader, self.task_count)
		return self.model
