import torch


def feat(network, xs):
    *layers, last = network
    for l in layers:
        xs = l(xs)
    return xs


def H_d(ps):
    ps_norm = ps / ps.sum()
    return - np.sum(np.log(ps) * ps_norm)


@torch.no_grad()
def H_srank(xs, feat_fn, net):
    zs = feat_fn(net, xs)
    gram_matrix = zs @ zs.T
    sgv = torch.linalg.svdvals(gram_matrix)
    sgv /= sgv.sum()
    return H_d(sgv.cpu().numpy())
