import random

import torch


def forward_diffusion_process(x, noise_scheduler, device: str = "cuda", num_timesteps: int = 1000):
    bs = x.shape[0]
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)

    noise = torch.randn_like(x)
    random_t_index = random.randint(0, num_timesteps - 1)
    t = noise_scheduler.timesteps[random_t_index]
    t_batch = torch.full(
        size=(x.shape[0],), 
        fill_value=t, 
        dtype=torch.long
    ).to(device)
    
    noisy_x = noise_scheduler.add_noise(x, noise, t_batch)

    return noisy_x, random_t_index


def backward_diffusion_process(x, t, y, model, noise_scheduler, device: str = "cuda", num_timesteps: int = 1000):
    bs = x.shape[0]
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)
    
    for t in noise_scheduler.timesteps[-t - 1:]:
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(bs,), 
            fill_value=t.item(), 
            dtype=torch.long
        ).to(device)

        noise_pred = model(
            model_input, t_batch, y, return_dict=False
        )[0]

        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x


def one_step_backward_diffusion_process(x, t, y, model, noise_scheduler, device: str = "cuda", num_timesteps: int = 1000):
    bs = x.shape[0]
    noise_scheduler.set_timesteps(num_inference_steps=num_timesteps)
    
    model_input = noise_scheduler.scale_model_input(x, torch.tensor([t]))

    t_batch = torch.full(
        size=(bs,), 
        fill_value=t, 
        dtype=torch.long
    ).to(device)

    noise_pred = model(
        model_input, t_batch, y, return_dict=False
    )[0]

    x = noise_scheduler.step(noise_pred, torch.tensor([t]), x).pred_original_sample

    return x
