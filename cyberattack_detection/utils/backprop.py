import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset



def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
    model.to('cuda')
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.FloatTensor(data)
    dataset = TensorDataset(data_x, data_x)
    bs = model.batch
    if training:
      dataloader = DataLoader(dataset, batch_size = bs, shuffle = True)
    else:
      dataloader = DataLoader(dataset, batch_size = bs, shuffle = False)
    l1s, l2s = [], []
    if training:
        model.train()
        for d, _ in dataloader:
            optimizer.zero_grad()
            d = d.to('cuda')
            window = d
            window = window.to('cuda')
            #z,mu, logvar = model(window,window)
            z = model(window,window)
            l1 = l(z, window)
            loss = torch.mean(l1)
            l1s.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')

        return np.mean(l1s), optimizer.param_groups[0]['lr']
    else:
        with torch.no_grad():
            model.eval()
            l1s = []
            recons = []
            for d, _ in dataloader:
                d = d.to('cuda')

                window = d
                window = window.to('cuda')
                z = model(window,window)
                l1 = l(z, window)
                recons.append(z)
                l1s.append(l1)
        return torch.cat(l1s).detach().cpu().numpy(), torch.cat(recons).detach().cpu().numpy()