
def download(url, savepath):
    import wget
    wget.download(url, str(savepath))


def unzip(zippath, savepath):
    import zipfile
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def fix_random_seed_as(random_seed):
    import numpy as np
    import random
    import torch
    import torch.backends.cudnn as cudnn
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False