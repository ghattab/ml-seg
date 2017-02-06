#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans

# create image patches from image directory
# clusters patches using Kmeans 

# retrieves image filenames given a path, returns list of filenames
def get_imgs(path):
    i_ext = tuple([".tif",".jpg",".jpeg",".png"])
    f = [os.path.join(path,f) for f in os.listdir(path) \
         if f.lower().endswith(i_ext)]
    ext = str.split(f[0],".")[-1] # recovered extension
    return f

# loads images from list of filenames, returns list of images and image shape
def load_imgs(f):
    rgb_imgs = []
    for i in f:
        rgb_imgs.append(cv2.imread(i, -1))
    size_var = rgb_imgs[0].shape
    return rgb_imgs, size_var

# directory with images of interest
path="./rgb_dir/"
f = get_imgs(path)
imgs, size = load_imgs(f)

# set dimensions of image patches n*n
n = 32
xs=range(0, size[0], n)
ys=range(0, size[1], n)
patches, filenames = [], []

for i in tqdm(range(len(imgs))):
    a = imgs[i]
    for x in xs:
        for y in ys:
            b = a[x:x + n, y:y + n,:]
            if b.shape == (n, n, 3):
                patches.append(np.ndarray.flatten(b))
                fn = str(i)+"p"+str(x)+"_"+str(y)+".tif"
                filenames.append(fn)
                # write image patches to disk
                a=cv2.imwrite(fn, imgs[i][x:x + n, y:y + n,:])


# cluster using Kmeans and return indices in variable
kmeans = KMeans(n_clusters=128, random_state=0).fit_predict(patches)

# output handling
filename = "../results.txt"
dir = os.path.dirname(filename)
if not os.path.exists(dir):
    os.makedirs(dir)

f = open(filename, "w")
for m in range(len(filenames)):
    # write m lines in the following format: 'filename.extension cluster_number' 
    f.write(str(filenames[m])+" "+str(kmeans[m])+"\n")

f.close()
