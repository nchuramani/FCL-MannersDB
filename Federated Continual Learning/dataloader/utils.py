import pandas as pd
import os
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

def ac_split(path):
	data = load_images(path)
	arrow = data[data['Using arrow'] == 1]
	circle = data[data['Using circle'] == 1]
	return arrow, circle