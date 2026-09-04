import torch
import numpy as np

# =============================================
# Loss Functions
# =============================================
def wasserstein_distance(Yhat: torch.Tensor, Y: torch.Tensor, p: int = 1, lambda_mu: float = 1, lambda_vol: float = 1) -> torch.Tensor:
    """
    compute difference of distributions in geometry of space, w1 OT solution plus moment regulerisation
    """
    # wp loss
    Yhat_sort = torch.sort(Yhat, dim = 0)[0]
    Y_sort = torch.sort(Y, dim = 0)[0]
    wp = torch.mean(torch.abs(Yhat_sort - Y_sort).pow(p)).pow(1.0/p)
    # moments reguleriser
    if Yhat.shape[0] > 1 and Y.shape[0] > 1:
        # asset moments
        mu_Yhat, mu_Y = torch.mean(Yhat, dim = 0), torch.mean(Y, dim = 0)
        std_Yhat, std_Y = torch.std(Yhat, dim = 0), torch.std(Y, dim = 0)
    # penalise moment deviation
    mu_reg = torch.abs(mu_Yhat - mu_Y)
    vol_reg = torch.abs(std_Yhat - std_Y)

    return wp + lambda_mu*mu_reg + lambda_vol*vol_reg

def mmd(X: torch.Tensor, Y: torch.Tensor, weighted_rbf = True, device = None):
    """
    (un)biased empirical mmd using RBF kernel, creates gram matrices (n, n) if X and Y are shape (n, m)
    """
    def rbf_kernel(x, sigma):
        return torch.exp(-x**2/(2*sigma**2))

    n, m = X.shape[0], Y.shape[0]
    assert n == m
    # make sure we have 1dim arrays
    if X.ndim > 1:
        X = X.flatten()
    if Y.ndim > 1:
        Y = Y.flatten()
    # bandwidth (median of offset matrix)
    a, b = torch.meshgrid(X, Y, indexing = 'ij')
    l2_diff = torch.pow(a - b, 2)
    i, j = torch.triu_indices(n ,n, offset = 1, device = device) # drop all 0 diagonal
    sigma = torch.sqrt(torch.median(l2_diff[i, j]))
    bandwidths = [sigma]
    # reshap
    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    # diff matrices
    xx = X.unsqueeze(1) - X.unsqueeze(0)
    yy = Y.unsqueeze(1) - Y.unsqueeze(0)
    xy = X.unsqueeze(1) - Y.unsqueeze(0)
    # rbf and weighetd rbf kernel
    rbf = lambda x, bands: torch.stack(
        [rbf_kernel(x, s) for s in bands]
    ).mean(dim=0)
    if weighted_rbf:
        bandwidths = [sigma*0.25, sigma, sigma*1.75]
    # gram matrices
    kxx = rbf(xx, bandwidths)
    kyy = rbf(yy, bandwidths)
    kxy = rbf(xy, bandwidths)
    # fill unbiased terms
    mask_x = torch.eye(n, dtype = torch.bool, device = kxx.device)
    mask_y = torch.eye(n, dtype = torch.bool, device = kyy.device) # identity matrix where 1=True
    kxx = kxx.masked_fill(mask_x, 0.0).to(device) # set all True to 0
    kyy = kyy.masked_fill(mask_y, 0.0).to(device)
    # similar terms
    sim_x = kxx.sum()
    sim_y = kyy.sum()
    # cross terms
    cross_xy = kxy.sum()

    return 1/(n**2) * (sim_x + sim_y) - 2/(n**2) * cross_xy

def marginal_kernel_density(Xp, X, h, use_reg = False, h_reg = 0.05):
    """
    kernel denisty estimate of point Xp under denisty X. This is scaled guassian expectation of Xp over X~p(x)

    h_reg:float = standard deviation of additive noise of X, X = X + z, z ~ N(0, eta), typicaly < h
    """
    eps = 1e-8
    # data
    Xp = Xp.unsqueeze(1)
    X = X.unsqueeze(0)
    # kernel
    diff = Xp - X
    diff_sq = diff.pow(2)
    w = torch.exp(- diff_sq / (2*h**2))
    prob_d = w / (h * np.sqrt(2 * np.pi))
    kernel_f = prob_d.mean(dim = 1) + eps # density of each point under X, add eps for numerical stability
    # noise reguliser
    reg = torch.tensor(0, device = Xp.device, dtype = Xp.dtype)
    if use_reg:
        wi = w / (w.sum(dim = 1, keepdim = True) + eps)
        second_diag = wi/h_reg**2 - 1/h_reg**4 * (1 - wi) * wi * diff_sq
        reg = second_diag.sum(dim = 1)

    return kernel_f, reg

def kl_divergance(Y:torch.Tensor, Yhat:torch.Tensor, reg_loss:bool = True, noise_reg: float = 1.0) -> torch.Tensor:
    """
    D_KL(p(y) || q(y)), forward kl divergance in continous
    """
    # dynamic bandwidth of kde variable
    unb = True
    n = Y.shape[0]
    if n <= 1:
        unb = False
    h = torch.std(Y, dim = -1, unbiased = unb) * 3
    eta = noise_reg * h

    # kde, kl and regularisation
    p_y, _ = marginal_kernel_density(Y, Y, h)
    q_yhat, kl_reg = marginal_kernel_density(Y, Yhat, h, reg_loss, eta)

    exp_f = torch.log(p_y/q_yhat)

    dkl = exp_f.mean() # expectation over each point, to get kernel for each point
    if reg_loss:
        dkl += kl_reg.mean() # broadcast (dkl) shape is [] so kl_reg must be shape []

    return dkl # in discrete setting expectation over y have density 1/n

def sharpe_ratio_loss(Y: torch.Tensor):
    """
    sharpe ratio loss function for dls model
    """
    eps = 1e-8
    unb = True
    if Y.shape[0] <= 1:
        unb = False
    exp_rt = torch.mean(Y, dim = 0)
    std_rt = torch.std(Y, dim = 0, unbiased = unb).clamp_min(eps)

    return -exp_rt / std_rt