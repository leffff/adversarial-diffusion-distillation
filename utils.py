import os
import random

importnumpy as np
import torch
import torchvision
from PIL import Image


def add_zero_class(y):
    bs = y.shape[0]

    y = (y + 1) * (torch.rand((bs,)) >= 0.1).long().to(y.device)
    return y


def seed_everything(seed: int,
                    use_deterministic_algos: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)
