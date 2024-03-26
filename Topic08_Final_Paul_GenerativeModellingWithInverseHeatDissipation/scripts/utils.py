import torch
import numpy as np
import random
import os
import logging
import cv2
from pathlib import Path
from PIL import Image
from torch.optim import Adam
import numpy as np
from torchvision.utils import make_grid, save_image

from models.unet import UNetModel
from models.ema import ExponentialMovingAverage


def safe_state(seed, device):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)    
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True


def restore_checkpoint(ckpt_dir, state, device):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if not os.path.exists(ckpt_dir):
        Path(os.path.dirname(ckpt_dir)).mkdir(parents=True, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['step'] = loaded_state['step']
        if 'ema' in state:
            state['ema'].load_state_dict(loaded_state['ema'])
        return state


def load_model(ckpt_dir, model, device):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if not os.path.exists(ckpt_dir):
        Path(os.path.dirname(ckpt_dir)).mkdir(parents=True, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return None
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        return loaded_state['model']


def save_checkpoint(ckpt_dir, state):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if 'ema' in state:
        saved_state = {
            'optimizer': state['optimizer'].state_dict(),
            'model': state['model'].state_dict(),
            'step': state['step'],
            'ema': state['ema'].state_dict()
        }
    else:
        saved_state = {
            'optimizer': state['optimizer'].state_dict(),
            'model': state['model'].state_dict(),
            'step': state['step']
        }
    torch.save(saved_state, ckpt_dir)


def save_video(save_dir, samples):
    """ Saves a video from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the video"""
    padding = 0
    nrow = int(np.sqrt(samples[0].shape[0]))
    imgs = []
    for idx in range(len(samples)):
        sample = samples[idx].cpu().detach().numpy()
        sample = np.clip(sample * 255, 0, 255)
        image_grid = make_grid(torch.Tensor(sample), nrow, padding=padding).numpy().transpose(1, 2, 0).astype(np.uint8)
        image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
        imgs.append(image_grid)
    #video_size = tuple(reversed(tuple(5*s for s in imgs[0].shape[:2])))
    video_size = tuple(reversed(tuple(s for s in imgs[0].shape[:2])))
    writer = cv2.VideoWriter(os.path.join(save_dir, "process.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),
                             30, video_size)
    for i in range(len(imgs)):
        image = cv2.resize(imgs[i], video_size, fx=0,
                           fy=0, interpolation=cv2.INTER_CUBIC)
        writer.write(image)
    writer.release()


def save_gif(save_dir, samples, name="process.gif"):
    """ Saves a gif from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the gif"""
    nrow = int(np.sqrt(samples[0].shape[0]))
    imgs = []
    for idx in range(len(samples)):
        s = samples[idx].cpu().detach().numpy()[:36]
        s = np.clip(s * 255, 0, 255).astype(np.uint8)
        image_grid = make_grid(torch.Tensor(s), nrow, padding=2)
        im = Image.fromarray(image_grid.permute(
            1, 2, 0).to('cpu', torch.uint8).numpy())
        imgs.append(im)
    imgs[0].save(os.path.join(save_dir, name), save_all=True,
                 append_images=imgs[1:], duration=0.5, loop=0)


def save_tensor(save_dir, data, name):
    """ Saves a Pytorch Tensor to save_dir with the given name."""
    with open(os.path.join(save_dir, name), "wb") as fout:
        np.save(fout, data.cpu().numpy())


def save_number(save_dir, data, name):
    """ Saves the number in argument 'data' as a text file and a .np file."""
    with open(os.path.join(save_dir, name), "w") as fout:
        fout.write(str(data))
    with open(os.path.join(save_dir, name) + ".np", "wb") as fout:
        np.save(fout, data)


def save_tensor_list(save_dir, data_list, name):
    """Saves a list of Pytorch tensors to save_dir with name 'name'"""
    with open(os.path.join(save_dir, name), "wb") as fout:
        np.save(fout, np.array([d.cpu().detach().numpy() for d in data_list]))


def save_png(save_dir, data, name, nrow=None):
    """Save tensor 'data' as a PNG"""
    if nrow == None:
        nrow = int(np.sqrt(data.shape[0]))
    image_grid = make_grid(data, nrow, padding=2)
    with open(os.path.join(save_dir, name), "wb") as fout:
        save_image(image_grid, fout)


def load_model_from_checkpoint_dir(config, checkpoint_dir, device, suffix="_cifar10"):
    """Another input definition for the restore_checkpoint wrapper,
    without a specified checkpoint number. 
    Assumes that the folder has file "checkpoint.pth"
    """
    model = create_model(config, device)
    optimizer = get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint{suffix}.pth')
    state = restore_checkpoint(checkpoint_path, state, device=device)
    logging.info("Loaded model from {}".format(checkpoint_dir))
    model = state['model']
    state['ema'].copy_to(model.parameters())
    return model


def create_model(config, device=None):
    """Create the model."""
    model = UNetModel(config)
    model = model.to(device)
    return model


def get_model_fn(model, train=False):
    """A wrapper for using the model in eval or train mode"""
    def model_fn(x, fwd_steps, y=None):
        """Args:
                x: A mini-batch of input data.
                fwd_steps: A mini-batch of conditioning variables for different levels.
        """
        if not train:
            model.eval()
            return model(x, fwd_steps, y)
        else:
            model.train()
            return model(x, fwd_steps, y)
    return model_fn

def get_optimizer(config, params):
    """Returns an optimizer object based on `config`.
    Copied from https://github.com/yang-song/score_sde_pytorch"""
    if config.optim.optimizer == 'Adam':
        optimizer = Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer