import torch
import torch.nn as nn

'''' Example

trainD_prom,testD_prom = split_train_test_windows(torch_data, 30)'''

def create_windows(data, n_window):
	"""
	Split data into train and test windows based on the value in column '18'.

	Parameters:
	- data: torch.Tensor, input data with shape (num_samples, num_features)
	- n_window: int, size of each window

	Returns:
	- train_windows: torch.Tensor, tensor containing windows of train data
	- test_windows: torch.Tensor, tensor containing windows of test data
	"""
	train_windows = []
	test_windows = []
	start_idx = 0
	while start_idx < len(data) - n_window:
		window = data[start_idx:start_idx + n_window]
		label_values = window[:,-1]
		if torch.sum(label_values > 0) < 8: #torch.all(label_values < 0):
				train_windows.append(window[:,:7])
		else:
				test_windows.append(window[:,:7])
		start_idx += 7
	return torch.stack(train_windows),torch.stack(test_windows)