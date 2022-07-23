import os
import numpy as np

import torch


def tensor2image(inp, dataset_mean, dataset_std):
    """Преобразует PyTorch тензоры для использования в matplotlib.pyplot.imshow"""
    out = inp.cpu().detach().numpy().transpose((1, 2, 0))
    mean = np.array(dataset_mean)
    std = np.array(dataset_std)
    out = std * out + mean

    return np.clip(out, 0, 1)


def smooth1d(data, window_width):
    """Сглаживает данные усреднением по окну размера window_width"""
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


def save_state(save_folder, name, model, optimizer, trained_iters, losses, losses_save_interval):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    state = {
        'model': model,
        'model_state': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state': optimizer.state_dict(),
        'trained_iters': trained_iters,
        'losses': losses,
        'losses_save_interval' : losses_save_interval
        }
        
    state_save_folder = os.path.join(save_folder, name)
    if not os.path.exists(state_save_folder):
        os.mkdir(state_save_folder)
    torch.save(state, os.path.join(state_save_folder, "state.pth"))

    
def load_state(save_folder, name):
    return torch.load(os.path.join(save_folder, name, "state.pth"))
