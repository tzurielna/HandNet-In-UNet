from os.path import splitext
from os import listdir
from scipy.io import loadmat
from math import sqrt
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from PIL import Image


dir = 'TrainData/'


class Conv2d3d:
    
    def __init__(self, KK_adress: str):
        self.KK = loadmat(KK_adress)['KK']
        self.invKK = np.linalg.inv(self.KK)

    @classmethod
    def find_center(cls, img) -> (int,int):
        """
        This function finds the center of a form in a np-array. This function calculating expected value to know the center.
        param: img: a np-array.
        res: coordinate (int,int) of the center.
        """
        expected = np.array([ 0, 0 ]).astype(float)
        sum = np.sum(img.astype(float))
        for i, j in product(range(img.shape[0]), range(img.shape[1])):
            expected += np.array([i , j]) * float(img[i, j])
        return int(expected[0] / sum), int(expected[1] / sum)

    @classmethod
    def find_max(cls, img) -> (int,int):
        """
        This function finds the center of a form in a np-array (as find_center). This function calculating argmax value to know the center.
        param: img: a np-array.
        res: coordinate (int,int) of the center.
        """
        iarr, jarr = np.argmax(img, axis=0), np.argmax(img, axis=1)
        i = 0 if len(iarr[iarr > 0]) == 0 else int(np.bincount(iarr[iarr > 0]).argmax())
        j = 0 if len(jarr[jarr > 0]) == 0 else int(np.bincount(jarr[jarr > 0]).argmax())
        return i, j
    
    @classmethod
    def normalize(cls, img):
        """
        This function normalize image to scale [0,1] with type float.
        param: img: a np-array.
        res: normalized img.
        """
        img = img.astype(float)
        img = img - np.min(img)
        img = img / np.max(img)
        return img
    
    @classmethod
    def plot(cls, depth, mask, path):
        """
        This function plot the image 'depth' with points.
        param: depth: a np-array which depth[i,j] is the distance for all i,j.
                      shape - w x h
        param: mask: a np-array.
                      shape - w x h x c
                              where c is the number of points.
        param: path: where to save the file (with suffix)
        res: none.
        """
        plt.clf()
        depth = Conv2d3d.normalize(depth)
        result = Image.fromarray((np.expand_dims(depth, axis=2).repeat(3, axis=2) * 255).astype(np.uint8))
        plt.imshow(result)
        w, h, c = mask.shape
        for k in range(c):
            i, j = Conv2d3d.find_center(mask[:,:,k])
            plt.scatter([j], [i])
        plt.savefig(path)
    
    def iup(self, depth, i, j):
        """
        This function calculate 3d point from 2d point, using depth array.
        param: depth: a np-array which depth[i,j] is the distance for all i,j.
        param: i,j: 2d point coordinate.
        res: coordinate (float,float,float) of the point.
        """
        depth_val = 0
        for k in range(10):
            d = depth[i-k:i+k+1, j-k:j+k+1].astype(float)
            d = d[d > 0]
            if len(d) == 0:
                continue
            depth_val = float(np.average(d))
            break
        line = np.dot(self.invKK, np.array([ i , j , 1 ]))
        x = line[0] / line[2]
        y = line[1] / line[2]
        a, b, c, d = self.KK[0,0], self.KK[1,1], self.KK[0,2], self.KK[1,2]
        x, y = (b * y + d - c) / a, (a * x + c - d) / b
        z = depth_val / sqrt(1 + x * x + y * y)
        return np.array([ x * z , y * z , z ])
    
    def up(self, depth, img):
        """
        This function calculate 3d point from image, using depth array.
        param: depth: a np-array which depth[i,j] is the distance for all i,j.
        param: img: a np-array (recommended a binary np-array).
        res: coordinate (float,float,float) of the point.
        """
        i, j = Conv2d3d.find_max(img)
        return self.iup(depth, i, j)
    

if __name__ == '__main__':
    convertor = Conv2d3d('Parameters.mat')
    ids = [splitext(file)[0] for file in listdir(dir) if not file.startswith('.')]
    sum_error, num_error = 0, 0
    for idx in ids:
        file = loadmat(dir + idx + '.mat')
        true_points = file['pos']
        hmap = file['hmap']
        depth = file['depth']
        num_classes = hmap.shape[2]
        error = np.array([np.linalg.norm(true_points[:,i] - convertor.up(depth, hmap[:,:,i])) for i in range(num_classes)])
        sum_error += np.average(error)
        num_error += 1
        print('average error:', sum_error / num_error, end='\r')
    print('total average error:', sum_error / num_error)
