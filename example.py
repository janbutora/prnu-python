# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018
"""
"""
Update December 2023
@author: Jan Butora (jan.butora@univ-lille.fr)
@author: Patrick Bas (patrick.bas@univ-lille.fr)
CNRS, University of Lille 2023
"""

import os
from glob import glob
from multiprocessing import cpu_count, Pool

import numpy as np
from PIL import Image

import prnu
import matplotlib.pyplot as plt


def main():
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    path = "data/LeicaQ2/"
    CROP_SIZE = 2048
    REMOVE_DIMPLES = True
    REMOVE_WATERMARK = False
    WATERMARK_SIZE = 128

    device = np.array(sorted(glob(path+'*')))
    device = np.array([os.path.split(i)[1] for i in device])
    ff_dirlist = np.array([np.array(sorted(glob(path+i+'/reference/*.jpg')+glob(path+i+'/reference/*.tif')),dtype=object) for i in device],dtype=object)
    nat_dirlist = np.array([np.array(sorted(glob(path+i+'/test/*.jpg')+glob(path+i+'/test/*.tif')),dtype=object) for i in device],dtype=object)
    nat_dirlist = np.array([path for source_list in nat_dirlist for path in source_list ])
    ff_device = np.array([path.split('__')[0].split('/')[-1] for source_list in ff_dirlist for path in source_list ])
    nat_device = np.array([path.split('__')[0].split('/')[-1] for path in nat_dirlist ])

    print('Computing fingerprints')
    fingerprint_device = sorted(np.unique(ff_device))
    k = []
    for source_list in ff_dirlist:
        imgs = []
        for img_path in source_list:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue
            im_cut = im_arr[:CROP_SIZE, :CROP_SIZE]
            imgs += [im_cut]
        k += [prnu.extract_multiple_aligned(imgs, dimples=REMOVE_DIMPLES, watermarked=REMOVE_WATERMARK, ws=WATERMARK_SIZE, processes=cpu_count())]

    k = np.stack(k, 0)

    print('Computing residuals')
    imgs = []
    for img_path in nat_dirlist:
        imgs += [np.asarray(Image.open(img_path))[:CROP_SIZE,:CROP_SIZE]]

    pool = Pool(cpu_count())
    w = pool.map(prnu.extract_single, imgs)
    pool.close()
    imgs = []
    w = np.stack(w, 0)

    # Computing Ground Truth
    gt = prnu.gt(fingerprint_device, nat_device)

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            img_path = nat_dirlist[natural_idx]
            img = prnu.rgb2gray(np.asarray(Image.open(img_path)))[:fingerprint_k.shape[0],:fingerprint_k.shape[1]]
            cc2d = prnu.crosscorr_2d(natural_w, fingerprint_k*img)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']


    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)

    if REMOVE_WATERMARK:
        print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.83))
    else:
        print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.68))

    plt.figure(figsize=(10,10),facecolor='white')
    pce_rot[pce_rot<1] = 1
    for i in range(len(device)):
        name = device[i]
        idx = np.array([name  for _ in range(len(pce_rot[i]))], dtype=str)
        plt.yscale("log")
        plt.grid()
        plt.scatter(idx[gt[i]==True],pce_rot[i,gt[i]==True],marker='x',linewidths=1, color='g')
        plt.scatter(idx[gt[i]==False],pce_rot[i,gt[i]==False],marker='+',linewidths=1,color='r')
    plt.axhline(y=60, color='r', linestyle='--')
    plt.xticks(rotation=20)
    plt.ylabel('PCE value')
    if REMOVE_WATERMARK:
        plt.savefig("Q2_PCE_without_watermark.pdf", bbox_inches='tight')
    else:
        plt.savefig("Q2_PCE.pdf", bbox_inches='tight')



if __name__ == '__main__':
    main()
