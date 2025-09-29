def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:  # Exclude frozen parameters
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            # Parameters to exclude from weight decay
            no_decay.append(param)
        else:
            # Parameters to apply weight decay
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},  # No weight decay
        {"params": decay, "weight_decay": weight_decay},  # Apply weight decay
    ]
