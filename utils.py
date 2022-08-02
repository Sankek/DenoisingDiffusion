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


def generate_images(model, variance_schedule, num=1, image_size=128, channels=3, device='cpu', reproducible=False, reproducible_seed=0):
    model.eval()
    shape = [num, channels, image_size, image_size]
    if reproducible:
        old_rng_state = torch.get_rng_state()
        torch.manual_seed(reproducible_seed)
    x_t = torch.normal(torch.zeros(shape), torch.ones(shape)).to(device)
    Tmax = variance_schedule.Tmax
    for t in range(Tmax-1, -1, -1):
        with torch.no_grad():
            pred_noise = model(x_t, torch.full((num,), t).to(device))
            
        alpha = variance_schedule.alpha[t]
        variance = variance_schedule.variance[t]
        alpha_prod_inv_sqrt = variance_schedule.alpha_prod_inv_sqrt[t]

        if t>0:
            z = torch.normal(torch.zeros(shape), torch.ones(shape)).to(device)
        else:
            z = 0
        
        x_t = 1/alpha**(1/2) * (x_t - variance/alpha_prod_inv_sqrt * pred_noise) + variance**(1/2) * z

    if reproducible:
        torch.set_rng_state(old_rng_state)
        
    return x_t.to('cpu')
