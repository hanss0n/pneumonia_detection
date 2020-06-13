import kaggle


def load_kaggle():
    dataset = 'paultimothymooney/chest-xray-pneumonia'
    target = 'kaggle'
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset=dataset, path=target, unzip=True, quiet=False)


if __name__ == '__main__':
    load_kaggle()