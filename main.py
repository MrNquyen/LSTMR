import torch
import random
import numpy as np

from utils.flags import Flags
from utils.configs import Config
from utils.utils import load_yml
from project.models.lstmr import LSTMR
# from utils.trainer import Trainer
from utils.trainer_modify_epoch import Trainer


def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across PyTorch, NumPy, and Python's random module."""
    np.random.seed(seed)  # NumPy seed
    random.seed(seed)    # Python's random module seed
    torch.manual_seed(seed) # PyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # PyTorch CUDA seed for current GPU
        torch.cuda.manual_seed_all(seed)


if __name__=="__main__":

    set_seed()

    # -- Get Parser
    flag = Flags()
    args = flag.get_parser()

    # -- Get device
    # print(args)
    config = args.config
    device = args.device

    # -- Get config
    config = load_yml(args.config)
    config_container = Config(config)

    # -- Trainer
    trainer = Trainer(config, args)
    if args.run_type=="train":
        trainer.train()
    elif args.run_type=="inference":
        trainer.inference(mode="test", save_dir="/datastore/npl/ViInfographicCaps/workspace/baseline/LSTMR/save/results")