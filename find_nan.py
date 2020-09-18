from scipy.io import loadmat
from os.path import splitext
from os import listdir
from numpy import min as check_nan
from math import isnan

imgs_dir = 'TrainData/'

if __name__ == '__main__':
    ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
    for idx in ids:
        file = loadmat(imgs_dir + idx)
        check1_nan = isnan(check_nan(file['hmap']))
        check2_nan = isnan(check_nan(file['ir']))
        if check1_nan or check2_nan:
            print(idx)

