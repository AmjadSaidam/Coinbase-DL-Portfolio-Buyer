import torch 

def min_rebalance(weights: torch.Tensor, minimum_value: float, iterations = 10) -> torch.Tensor:
    """ 
    """
    # clone_weights so we do not detach from computation graph 
    weights = weights.clone()

    # deal with edge case 
    elements = weights.numel()
    if minimum_value * elements > 1:
        raise ValueError('minvalue*elements > 1, constraint is infeasible, reduce minimum value')

    # 1
    new_weights = torch.clamp(weights, min = minimum_value)

    # 2
    new_weight_cond = new_weights <= minimum_value + 1e-12
    new_weights_set = new_weights[new_weight_cond] # all floored weights
    invarient_weights_set = new_weights[~new_weight_cond] # boolean mask 

    # 3
    new_weight_set_diff = torch.sum(minimum_value - weights[new_weight_cond])

    # 4
    sum_invarient_set = torch.sum(invarient_weights_set)
    if new_weight_set_diff.item() > 0:
        new_weights[~new_weight_cond] -= (new_weights[~new_weight_cond] / sum_invarient_set)*new_weight_set_diff # will edit new_weights in place

    # last normalisation
    w = torch.clamp(new_weights, min=minimum_value)
    w /= torch.sum(w)

    # recursion 
    if iterations > 0 and torch.any(w < minimum_value): # stoping case
        # repeat process 
        return min_rebalance(weights = w, minimum_value = minimum_value, iterations = iterations - 1)
    else: 
        return w