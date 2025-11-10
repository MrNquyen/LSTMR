
from bisect import bisect
def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    return parameters


# def lr_lambda_update(i_iter, cfg):
#     if (
#         cfg["use_warmup"] is True
#         and i_iter <= cfg["warmup_iterations"]
#     ):
#         alpha = float(i_iter) / float(cfg["warmup_iterations"])
#         return cfg["warmup_factor"] * (1.0 - alpha) + alpha
#     else:
#         idx = bisect(cfg["lr_steps"], i_iter)
#         return pow(cfg["lr_ratio"], idx)


def lr_lambda_update(current_epoch, cfg):
    """
    Learning rate scheduler with exponential decay
    
    Configuration for the specific setup:
    - Initial LR: 2e-4 (set in optimizer config)
    - Decay every 3 epochs with factor 0.8
    - Formula: new_lr = initial_lr * (decay_factor ** num_decays)
    
    Args:
        current_epoch: Current training epoch
        cfg: Config dictionary containing:
            - decay_factor: 0.8 (annealing factor)
            - step_epoch_size: 3 (decay every 3 epochs)
    
    Returns:
        Learning rate multiplier
    """
    decay_factor = cfg.get("decay_factor", 0.8)  # Default annealing factor 0.8
    step_epoch_size = cfg.get("step_epoch_size", 3)  # Default decay every 3 epochs
    
    num_decays = current_epoch // step_epoch_size
    return decay_factor ** num_decays
