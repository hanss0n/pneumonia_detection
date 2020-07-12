import numpy as np
import torch


# TODO: move each augmentation method to its own class

# DISCLAIMER: Code for mixup-implementation from: https://www.kaggle.com/qqgeogor/keras-nn-mixup/comments#480542
def mixup(x, y, alpha=1.0, seed=1337):
    if seed is not None:
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
    if seed is not None:
        np.random.seed(seed)
    batch_size = len(x)
    indices = np.random.permutation(batch_size)
    shuffled_data = x[indices]

    lam = np.random.beta(alpha, alpha)
    lam = np.array(lam)
    bbx1, bby1, bbx2, bby2 = __rand_bbox(x.shape, lam, seed=seed)
    x[:, bbx1:bbx2, bby1:bby2, :] = shuffled_data[:, bbx1:bbx2, bby1:bby2, :]

    return x, y


def __rand_bbox(size, lam, seed=1337):
    if seed is not None:
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


# DISCLAIMER: Code inspired by: https://github.com/changewOw/Cutout-numpy
def cutout(x_cut, y_cut, n_holes=10, seed=1337, img_height=150, img_width=150, max_height=15.0, min_height=5.0,
           max_width=15.0, min_width=5.0):
    if seed is not None:
        np.random.seed(seed)

    shuffled_data = np.ones((img_height, img_width, 1), dtype=np.int32)
    # TODO: add random color fill/hole
    shuffled_data.fill(0)

    for n in range(n_holes):
        y = np.random.randint(img_height)
        x = np.random.randint(img_width)

        h_l = np.random.randint(min_height, max_height + 1)
        w_l = np.random.randint(min_width, max_width + 1)

        y1 = np.clip(y - h_l // 2, 0, img_height)
        y2 = np.clip(y + h_l // 2, 0, img_height)
        x1 = np.clip(x - w_l // 2, 0, img_width)
        x2 = np.clip(x + w_l // 2, 0, img_width)
        x_cut[:, x1:x2, y1:y2, :] = shuffled_data[x1:x2, y1:y2]

    return x_cut, y_cut
