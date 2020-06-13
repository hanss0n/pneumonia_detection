import kaggle
import os


def load_kaggle():
    if not os.path.exists(os.path.join('kaggle', 'chest_xray')):
        dataset = 'paultimothymooney/chest-xray-pneumonia'
        target = 'kaggle'
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=dataset, path=target, unzip=True, quiet=False)
    else:
        print('The Kaggle dataset is already downloaded')


if __name__ == '__main__':
    load_kaggle()
