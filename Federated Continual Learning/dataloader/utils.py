import pandas as pd
import os
import torchvision.transforms as transforms
from .imageloader import CustomDataset
from torch.utils.data import random_split, DataLoader
from .dataloader import MyDataset
import torch
import numpy as np
import sys

sys.path.append('..')
from utils import predict_gen, train, predict_gen_distil


# Try "cuda" to train on GPU


def load_universal(path):
	data = pd.read_csv(path + "/all_data.csv")
	df = pd.DataFrame(columns=data.columns)
	for i in range(250, 1000):
		df = pd.concat([df, (data.loc[data['Stamp'] == i].mean()).to_frame().T], ignore_index=True)
	# df=df.append(data.loc[data['Stamp'] == i].mean(), ignore_index=True)
	return df


def gaussian_noise(x, mu, std):
	noise = np.random.normal(mu, std, size=x.shape)
	for i in range(3, 30):
		x[i] = x[i] + noise[i]
	return x


def load_augmented(df):
	df_input = df.iloc[:, 0:30]
	df_labels = df.iloc[:, 30:]
	df_input_new = pd.DataFrame(columns=df.columns[:30])
	df_labels_new = pd.DataFrame(columns=df_labels.columns)
	for c in range(25):
		for i in range(df_input.shape[0]):
			df_input_new = pd.concat([df_input_new, gaussian_noise(df_input.iloc[i], 0, 0.01).to_frame().T])
			df_labels_new = pd.concat([df_labels_new, df_labels.iloc[i].to_frame().T])
	data_new = pd.concat([df_input_new, df_labels.reindex(df_input_new.index)], axis=1)
	
	return data_new


def a_c_split_pandas(path):
	df = load_universal(path)
	arrow = df[df['Using arrow'] == 1]
	circle = df[df['Using circle'] == 1]
	return arrow, circle


def splitter(path, n_clients, aug):
	arrow, circle = a_c_split_pandas(path)
	
	arrow = arrow.sample(frac=1).reset_index(drop=True)
	circle = circle.sample(frac=1).reset_index(drop=True)
	
	train_arrow_cl = []
	train_circle_cl = []
	y_labels = arrow.columns[-8:]
	
	test_circle = circle.iloc[int(circle.shape[0] * 0.75):]
	test_circle = DataLoader(MyDataset(np.array(test_circle.iloc[:, 3:-8]), np.array(test_circle.iloc[:, -8:])), batch_size=16, drop_last=True)
	
	test_arrow = arrow.iloc[int(arrow.shape[0] * 0.75):]
	test_arrow = DataLoader(MyDataset(np.array(test_arrow.iloc[:, 3:-8]), np.array(test_arrow.iloc[:, -8:])), batch_size=16, drop_last=True)
	
	test_combined = pd.concat([circle.iloc[int(circle.shape[0] * 0.75):], arrow.iloc[int(arrow.shape[0] * 0.75):]], axis=0)
	test_combined = DataLoader(MyDataset(np.array(test_combined.iloc[:, 3:-8]), np.array(test_combined.iloc[:, -8:])), batch_size=16, drop_last=True)
	
	arrow = arrow.iloc[:int(arrow.shape[0] * 0.75)]
	circle = circle.iloc[:int(circle.shape[0] * 0.75)]
	if aug:
		arrow = load_augmented(arrow)
		circle = load_augmented(circle)
		
		arrow = arrow.sample(frac=1).reset_index(drop=True)
		circle = circle.sample(frac=1).reset_index(drop=True)
	
	arrow_size = int(arrow.shape[0] / n_clients)
	circle_size = int(circle.shape[0] / n_clients)
	
	for i in range(n_clients):
		train_arrow_cl.append(DataLoader(MyDataset(np.array(arrow.iloc[i * arrow_size:(i + 1) * arrow_size][arrow.columns[3:-8]]),
		                                           np.array(arrow.iloc[i * arrow_size:(i + 1) * arrow_size][arrow.columns[-8:]])),
		                                 batch_size=16, shuffle=True, drop_last=True))
		train_circle_cl.append(DataLoader(MyDataset(np.array(circle.iloc[i * circle_size:(i + 1) * circle_size][circle.columns[3:-8]]),
		                                            np.array(circle.iloc[i * circle_size:(i + 1) * circle_size][circle.columns[-8:]])),
		                                  batch_size=16, shuffle=True, drop_last=True))
	
	return [train_circle_cl, train_arrow_cl], [test_circle, test_arrow], test_combined, y_labels


def Average(lst):
	return sum(lst) / len(lst)


def load_images(path):
	df = load_universal(path)
	path = path + "/screenshots"
	li = os.listdir(path)
	data_images = pd.DataFrame(columns=['Stamp', 'path'] + list(df.columns[1:3]) + list(df.columns[-8:]))
	for i in range(len(li)):
		try:
			entry = [float(li[i][:3]),
			         f'{path}/' + li[i],
			         int(df[df['Stamp'] == float(li[i][:3])]['Using circle']),
			         int(df[df['Stamp'] == float(li[i][:3])]['Using arrow']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Vacuum cleaning']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Mopping the floor']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Carry warm food']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Carry cold food']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Carry drinks']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Carry small objects (plates, toys)']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Carry big objects (tables, chairs)']),
			         float(df[df['Stamp'] == float(li[i][:3])]['Cleaning (Picking up stuff) / Starting conversation'])
			         ]
			new_df = pd.DataFrame([entry], columns=data_images.columns)
			data_images = pd.concat([data_images, new_df], ignore_index=True)
		except:
			print('Error Loading FileName; Continuing to next.')
	
	return data_images


def load_datasets(num_clients, net, path, aug, batch_size=16, distil=False, out='', DEVICE=torch.device("cpu")):
	data = load_images(path)
	
	# Splitting 75% Train and 25% Test Data
	data_images_train = data.iloc[:int(data.shape[0] * 0.75)]
	data_images_test = data.iloc[int(data.shape[0] * 0.75):]
	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	else:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	trainset = CustomDataset(dataframe=data_images_train, transform=train_transform)
	testset = CustomDataset(dataframe=data_images_test, transform=test_transform)
	
	# Split training set into `num_clients` partitions to simulate different local datasets
	partition_size = len(trainset) // num_clients
	
	lengths = [partition_size] * num_clients
	# trim trainset to partition_size*num_clients without iloc
	trainset = torch.utils.data.Subset(trainset, range(partition_size * num_clients))
	datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
	y_labels = data.columns[-8:]
	
	if distil:
		if os.path.exists(f'{out}/teacher_model.pt'):
			teacher_model = net.to(DEVICE)
			teacher_model.load_state_dict(torch.load(f'{out}/teacher_model.pt'))
		elif os.path.exists(f'{out}/teacher_model_cuda.pt'):
			teacher_model = net.to(DEVICE)
			teacher_model.load_state_dict(torch.load(f'{out}/teacher_model_cuda.pt'))
		else:
			teacher_model = net.to(DEVICE)
			trainset = CustomDataset(dataframe=data_images_train, transform=train_transform)
			train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
			print("Training Teacher Model")
			train(model=teacher_model, train_loader=train_loader, epochs=10, DEVICE=DEVICE)
		if DEVICE.type == 'cuda':
			torch.save(teacher_model.state_dict(), f'{out}/teacher_model_cuda.pt')
		else:
			torch.save(teacher_model.state_dict(), f'{out}/teacher_model.pt')
	
	# Split each partition into train/val and create DataLoader
	trainloaders = []
	valloaders = []
	
	for ds in datasets:
		len_val = len(ds) // 10  # 10 % validation set
		len_train = len(ds) - len_val
		lengths = [len_train, len_val]
		ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
		if distil:
			trainloaders.append(
				predict_gen_distil(teacher_model, DataLoader(ds_train, batch_size=batch_size, shuffle=True), DEVICE))
		else:
			trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
		valloaders.append(DataLoader(ds_val, batch_size=batch_size))
	testloader = DataLoader(testset, batch_size=batch_size)
	return trainloaders, valloaders, testloader, y_labels


def ac_split(path):
	data = load_images(path)
	arrow = data[data['Using arrow'] == 1]
	circle = data[data['Using circle'] == 1]
	return arrow, circle


def task_splitter_circle_arrow(path, n_clients, aug, batch_size=16):
	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	else:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	arrow, circle = ac_split(path)
	train_arrow_cl = []
	train_circle_cl = []
	y_labels = arrow.columns[-8:]
	
	test_circle = circle.iloc[int(circle.shape[0] * 0.75):]
	test_circle = DataLoader(CustomDataset(test_circle, transform=test_transform), batch_size=batch_size)
	
	test_arrow = arrow.iloc[int(arrow.shape[0] * 0.75):]
	test_arrow = DataLoader(CustomDataset(test_arrow, transform=test_transform), batch_size=batch_size)
	
	test = pd.concat([arrow.iloc[int(arrow.shape[0] * 0.75):], circle.iloc[int(circle.shape[0] * 0.75):]], axis=0)
	
	test = DataLoader(CustomDataset(test, test_transform), batch_size=batch_size)
	
	arrow = arrow.iloc[:int(arrow.shape[0] * 0.75)]
	circle = circle.iloc[:int(circle.shape[0] * 0.75)]
	
	size = int(arrow.shape[0] / n_clients)
	
	for i in range(n_clients):
		train_arrow_cl.append(DataLoader(CustomDataset(arrow.iloc[i * size:(i + 1) * size], train_transform), batch_size=batch_size, shuffle=True))
		train_circle_cl.append(DataLoader(CustomDataset(circle.iloc[i * size:(i + 1) * size], train_transform), batch_size=batch_size, shuffle=True))
	
	return [train_arrow_cl, train_circle_cl], [test_arrow, test_circle], test, y_labels


def load_datasets_hyper(path, aug):
	data = load_images(path)
	data_images_train = data.iloc[:int(data.shape[0] * 0.75)]
	data_images_test = data.iloc[int(data.shape[0] * 0.75):]
	if aug == True:
		transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
		])
	else:
		transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize if required
		])
	trainset = CustomDataset(dataframe=data_images_train, transform=transform)
	testset = CustomDataset(dataframe=data_images_test, transform=transform)
	return trainset, testset
