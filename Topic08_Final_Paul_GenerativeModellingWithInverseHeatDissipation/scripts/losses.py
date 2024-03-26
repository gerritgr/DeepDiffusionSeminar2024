"""All functions related to loss computation
"""

import torch
import numpy as np


def optimization_manager(config):
    """Returns an optimize_fn based on `config`.
    Based on code from https://github.com/yang-song/score_sde_pytorch"""
    if config.optim.automatic_mp:
        def optimize_fn(optimizer, params, step, scaler, lr=config.optim.lr,
                        warmup=config.optim.warmup,
                        grad_clip=config.optim.grad_clip):
            """Optimizes with warmup and gradient clipping (disabled if negative).
            Before that, unscales the gradients to the regular range from the 
            scaled values for automatic mixed precision"""
            scaler.unscale_(optimizer)
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)
            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            # Since grads already scaled, this just takes care of possible NaN values
            scaler.step(optimizer)
            scaler.update()
    else:
        def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                        warmup=config.optim.warmup,
                        grad_clip=config.optim.grad_clip):
            """Optimizes with warmup and gradient clipping (disabled if negative)."""
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)
            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
    return optimize_fn


def get_label_sampling_function(K):
    return lambda batch_size, device: torch.randint(1, K, (batch_size,), device=device)


def get_step_fn(train, config, loss_fn=None, optimize_fn=None):
    """A wrapper for loss functions in training or evaluation
    Based on code from https://github.com/yang-song/score_sde_pytorch"""

    # For automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    def step_fn(state, x, y):
        """Running one step of training or evaluation.
        Returns:
                loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            if config.optim.automatic_mp:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, losses_batch, fwd_steps_batch = loss_fn(model, x, y)
                    # amp not recommended in backward pass, but had issues getting this to work without it
                    # Followed https://github.com/pytorch/pytorch/issues/37730
                    scaler.scale(loss).backward()
                scaler.scale(losses_batch)
                optimize_fn(optimizer, model.parameters(), step=state['step'], scaler=scaler)
                state['step'] += 1
                state['ema'].update(model.parameters())
            else:
                optimizer.zero_grad()
                loss, losses_batch, fwd_steps_batch = loss_fn(model, x, y)
                loss.backward()
                optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['step'] += 1
                state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss, losses_batch, fwd_steps_batch = loss_fn(model, x, y)
                ema.restore(model.parameters())

        return loss, losses_batch, fwd_steps_batch

    return step_fn
