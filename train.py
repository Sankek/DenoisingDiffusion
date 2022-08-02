import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

from utils import save_state, tensor2image, smooth1d, generate_images


def train_step_graph(generated_images, losses,
                     examples_suptitle_text='', losses_suptitle_text='', losses_smooth_window=25):
    num_examples = len(generated_images)
    
    fig, axs = plt.subplots(1, num_examples, figsize=(num_examples*3, 3), squeeze=False)
    for i in range(num_examples):
        axs[0, i].imshow(generated_images[i])
        axs[0, i].set_title('Generated Image')
        axs[0, i].axis('off')
    plt.suptitle(examples_suptitle_text)
    fig.tight_layout(pad=2)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(smooth1d(np.array(losses), losses_smooth_window), label='losses')
    ax.legend()
    plt.suptitle(losses_suptitle_text)
    plt.show()


def train(model, optimizer, dataloader, criterion, variance_schedule, dataset_mean, dataset_std,  
          epochs=1, graph_show_interval=10, losses_smooth_window=25, device='cpu',
          trained_iters=0, save_interval=10000, save_folder='.', save_name='baseline', grad_accum=1):
    
    batch_size = dataloader.batch_size

    losses = []
    optimizer.zero_grad()
    for epoch in range(epochs):
        for batch_num, (noised_img, t, noise) in enumerate(dataloader):
            model.train()
    
            noised_img = noised_img.to(device)
            t = t.to(device)
            noise = noise.to(device)
            
            pred_noise = model(noised_img, t)
            loss = criterion(pred_noise, noise)
            losses.append(loss.item())

            (loss/grad_accum).backward()
            if batch_num%grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            trained_iters += len(t)

            losses_save_interval = batch_size
            if (trained_iters % save_interval) < batch_size:
                save_state(save_folder, f'{save_name}_{trained_iters}', model, optimizer, trained_iters, losses, losses_save_interval)


            # Example images and losses graph 
            # -----------------------------------
            if batch_num % graph_show_interval == 0:
                losses_suptitle_text = f"{batch_num+1}/{len(dataloader)}"
                examples_suptitle_text = f""

                genims = generate_images(model, variance_schedule, num=5, device=device, reproducible=True)
                generated_images = [tensor2image(genim, dataset_mean, dataset_std) for genim in genims]
                
                clear_output(wait=True)
                train_step_graph(
                    generated_images, losses, 
                    examples_suptitle_text, losses_suptitle_text, losses_smooth_window
                )
            # -----------------------------------

    return losses
