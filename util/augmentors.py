import numpy as np
import matplotlib.pyplot as plt


# DISCLAIMER: Code for mixup-implementation from: https://www.kaggle.com/qqgeogor/keras-nn-mixup/comments#480542
def mixup(features, labels, alpha=1.0, show_sample=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = np.random.choice(np.arange(len(features)))
    old = features[index][:, :, 0].copy()

    sample_size = features.shape[0]
    index_array = np.arange(sample_size)
    np.random.shuffle(index_array)

    mixed_features = lam * features + (1 - lam) * features[index_array]
    mixed_labels = (lam * labels) + ((1 - lam) * labels[index_array])
    if show_sample:
        new = mixed_features[index][:, :, 0]
        __show_sample(old, new, 'mixup')
    return mixed_features, mixed_labels


# DISCLAIMER: Code for CutMix implementation inspired by: https://github.com/airplane2230/keras_cutmix
def cutmix(features, labels, alpha=1.0, show_sample=False):
    index = np.random.choice(np.arange(len(features)))
    old = features[index][:, :, 0].copy()
    batch_size = len(features)
    indices = np.random.permutation(batch_size)
    shuffled_data = features[indices]

    lam = np.random.beta(alpha, alpha)
    lam = np.array(lam)
    bbx1, bby1, bbx2, bby2 = __cutmix_box(features.shape, lam)
    features[:, bbx1:bbx2, bby1:bby2, :] = shuffled_data[:, bbx1:bbx2, bby1:bby2, :]

    if show_sample:
        new = features[index][:, :, 0]
        __show_sample(old, new, 'cutmix')

    return features, labels


# DISCLAIMER: Code inspired by: https://github.com/changewOw/Cutout-numpy
def cutout(features, labels, n_holes=1, img_height=150, img_width=150, max_height=15.0, min_height=5.0,
           max_width=15.0, min_width=5.0, show_sample=False):
    index = np.random.choice(np.arange(len(features)))
    old = features[index][:, :, 0].copy()
    shuffled_data = np.ones((img_height, img_width, 1), dtype='float32')
    for n in range(n_holes):
        shuffled_data, y1, y2, x1, x2 = __cutout_box(shuffled_data, img_height, img_width, max_height, min_height,
                                                     max_width, min_width)
        features[:, x1:x2, y1:y2, :] = shuffled_data[x1:x2, y1:y2]
    if show_sample:
        new = features[index][:, :, 0]
        __show_sample(old, new, 'cutout')
    return features, labels


def single_cutout(img, n_holes=3, img_height=150, img_width=150, max_height=20.0, min_height=5.0,
                  max_width=20.0, min_width=3.0):
    shuffled_data = np.ones((img_height, img_width, 1), dtype='float32')
    for n in range(n_holes):
        shuffled_data, y1, y2, x1, x2 = __cutout_box(shuffled_data, img_height, img_width, max_height, min_height,
                                                     max_width, min_width)
        img[x1:x2, y1:y2] = shuffled_data[x1:x2, y1:y2]
    return img


def __cutmix_box(size, lam):
    img_width = size[1]
    img_height = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(img_width * cut_rat)
    cut_h = np.int(img_height * cut_rat)

    # uniform
    cx = np.random.randint(img_width)
    cy = np.random.randint(img_height)

    bbx1 = np.clip(cx - cut_w // 2, 0, img_width)
    bby1 = np.clip(cy - cut_h // 2, 0, img_height)
    bbx2 = np.clip(cx + cut_w // 2, 0, img_width)
    bby2 = np.clip(cy + cut_h // 2, 0, img_height)

    return bbx1, bby1, bbx2, bby2


def __cutout_box(shuffled_data, img_height, img_width, max_height, min_height, max_width, min_width):
    shuffled_data.fill(np.random.uniform(0.0, 1.0))
    y = np.random.randint(img_height)
    x = np.random.randint(img_width)
    h_l = np.random.randint(min_height, max_height + 1)
    w_l = np.random.randint(min_width, max_width + 1)
    y1 = np.clip(y - h_l // 2, 0, img_height)
    y2 = np.clip(y + h_l // 2, 0, img_height)
    x1 = np.clip(x - w_l // 2, 0, img_width)
    x2 = np.clip(x + w_l // 2, 0, img_width)
    return shuffled_data, y1, y2, x1, x2


def __show_sample(old, new, aug_method):
    plt.imshow(old)
    plt.title('Before {}'.format(aug_method))
    plt.savefig('images/before_{}.png'.format(aug_method))
    plt.show()
    plt.imshow(new)
    plt.title('After {}'.format(aug_method))
    plt.savefig('images/after_{}.png'.format(aug_method))
    plt.show()
