import numpy as np


# DISCLAIMER: Code for mixup-implementation from: https://www.kaggle.com/qqgeogor/keras-nn-mixup/comments#480542
def mixup(x, y, alpha=1.0, seed=1337):
    np.random.seed(seed)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    sample_size = x.shape[0]
    index_array = np.arange(sample_size)
    np.random.shuffle(index_array)

    mixed_x = lam * x + (1 - lam) * x[index_array]
    mixed_y = (lam * y) + ((1 - lam) * y[index_array])
    return mixed_x, mixed_y


# DISCLAIMER: Code for CutMix implementation inspired by: https://github.com/airplane2230/keras_cutmix
def cutmix(x, y, alpha=1.0, seed=1337):
    np.random.seed(seed)
    batch_size = len(x)
    indices = np.random.permutation(batch_size)
    shuffled_data = x[indices]

    lam = np.random.beta(alpha, alpha)
    lam = np.array(lam)
    bbx1, bby1, bbx2, bby2 = __rand_bbox(x.shape, lam)
    x[:, bbx1:bbx2, bby1:bby2, :] = shuffled_data[:, bbx1:bbx2, bby1:bby2, :]

    return x, y


def __rand_bbox(size, lam, seed=1337):
    np.random.seed(seed)
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
