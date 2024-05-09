import torch

from diffusion import forward_diffusion_process, backward_diffusion_process


def train_epoch(S, T, D, S_optimizer, D_optimizer, accelerator, adversarial_loss, reconstruction_loss, lambd, dataloader, device: str = "cuda", num_student_timesteps: int = 4, num_teacher_timesteps: int = 1000):
    S.train()
    T.eval()
    D.train()

    S_losses = []
    D_losses = []
    
    for batch in tqdm(dataloader):
        x = batch["images"]
        labels = add_zero_class(batch["label"])
        
        noise = torch.randn_like(x).to(device)
        
        bs = x.shape[0]
        target_real = torch.ones((bs,), dtype=torch.long, device=device)
        target_fake = torch.zeros((bs,), dtype=torch.long, device=device)

        # Foward diffusion process on clean image
        xs, s = forward_diffusion_process(x, noise_scheduler, num_timesteps=num_student_timesteps)

        # Train ADD-student
        # Backward diffusion process on noised image, using noisy_images and t
        S_optimizer.zero_grad()
        x_theta = one_step_backward_diffusion_process(xs, s, labels, S, noise_scheduler, num_timesteps=num_student_timesteps)
        L_G_adv = adversarial_loss(D(x_theta), target_real)

        # Foward diffusion process on an image denoised by ADD-student
        xt, t = forward_diffusion_process(x_theta, noise_scheduler, num_timesteps=num_teacher_timesteps)
        with torch.no_grad():
            x_psi = one_step_backward_diffusion_process(xt, t, labels, T, noise_scheduler, num_timesteps=num_teacher_timesteps)

        с = 1 / (t + 1)
        d = reconstruction_loss(x_theta, x_psi) * с # * c(t), where c(t) = a_t

        S_loss = L_G_adv + lambd * d
                
        # accelerator.clip_grad_norm_(S.parameters(), 1.0)                     
        accelerator.backward(S_loss)
        S_optimizer.step()
        
        # Train Descriminator
        D_optimizer.zero_grad()
        real_loss = adversarial_loss(D(x), target_real) # TODO: Need R1 regularization
        fake_loss = adversarial_loss(D(x_theta.detach()), target_fake)
        L_D_adv = (real_loss + fake_loss) / 2
        
        # accelerator.clip_grad_norm_(D.parameters(), 1.0)
        accelerator.backward(L_D_adv)
        D_optimizer.step()

        S_losses.append(S_loss.item())
        D_losses.append(L_D_adv.item())

    S_losses, D_losses = sum(S_losses) / len(dataloader.dataset), sum(D_losses) / len(dataloader.dataset)

    return S_losses, D_losses


def sample_images(model, noise_scheduler, device: str = "cuda", c: int = 0, bs: int =16, num_inference_steps: int = 1000):
    model.train()
    model.to(device)

    x = torch.randn((bs, 3, 128, 128)).to(device)

    y_uncond = torch.zeros((bs,), device=device).long()
    y_cond = torch.ones((bs,), device=device).long() * c
    
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    for t in noise_scheduler.timesteps:
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(x.shape[0],), 
            fill_value=t.item(), 
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            noise_pred = model(
                model_input, 
                t_batch, 
                y_cond,
                return_dict=False
            )[0]

        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x


def sample_images_cfg(model, noise_scheduler, device: str = "cuda", c: int = 0, w: float = 1, bs: int = 16, num_inference_steps: int = 1000):
    model.train()
    model.to(device)

    x = torch.randn((bs, 3, 128, 128)).to(device)

    y_uncond = torch.zeros((bs,), device=device).long()
    y_cond = torch.ones((bs,), device=device).long() * c
    
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    for t in noise_scheduler.timesteps:
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(x.shape[0],), 
            fill_value=t.item(), 
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            cond_noise_pred = model(
                model_input, 
                t_batch, 
                y_cond,
                return_dict=False
            )[0]

            uncond_noise_pred = model(
                model_input, 
                t_batch, 
                y_uncond,
                return_dict=False
            )[0]

            noise_pred = (1 + w) * cond_noise_pred - w * uncond_noise_pred
            
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x
