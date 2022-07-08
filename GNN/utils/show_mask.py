from __future__ import print_function
from builtins import range

import sys
import numpy as np
import cv2
import pickle as pickle
import os
import glob
import matplotlib.pyplot as plt

def get_all(path, ext="pkl"):
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(path, size):
    if os.path.exists(path+".jpg"):
        p = path+".jpg"
    elif os.path.exists(path+".JPG"):
        p = path+".JPG"
    elif os.path.exists(path+".png"):
        p = path+".png"
    elif os.path.exists(path+".PNG"):
        p = path+".PNG"

    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image, size=size)
    image = image.astype(np.float32)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def resize(img, size=(1024,512)):
    return cv2.resize(img.astype('float32'), size).astype('int32')

def show_cells(drawing):
    """
    Show the image
    :return:
    """
    plt.imshow(drawing)
    plt.show()

def show_img(img1, img2, img_gt, dest):
    """, rows
    Show the image
    :return:
    """
    fig, ax = plt.subplots(2,2)
    # drawing -= drawing.min()
    # drawing /= drawing.max()
    ax[0,0].imshow(img1)
    ax[1,0].imshow(img2)
    ax[0,1].imshow(img_gt)
    # plt.show()
    plt.savefig(dest, format='eps', dpi=300)
    plt.close()

def create_img(cols, size, reverse=False):
    if type(cols) is not tuple:
        if not reverse:
            a =  np.array([cols]*size)
        else:
            a =  np.array([cols]*size).T
    else:
        cols_, starts = cols
        a = np.zeros((len(cols_), size))
        for i in range(len(cols_)):
            start = int(starts[i])
            num_pix = cols_[i]
            nums = int((num_pix*size))
            a[i, start:start+nums] = 1
        if not reverse:
            a = a.T
    return np.repeat(a[:, :, np.newaxis], 3, axis=2)

def main():
    """
    Quick script to show mask images stored on pickle files
    """
    pkl_dir = sys.argv[1]
    img_dir = sys.argv[2]
    rc_dir = sys.argv[3]
    dir_to_save = sys.argv[4]
    SKEW = True
    create_dir(dir_to_save)
    file_list = get_all(pkl_dir)
    row_skew, col_skew = None, None
    for dir_file in file_list:
        fname = dir_file.split("/")[-1].split(".")[0]
        fname_gt = os.path.join(rc_dir, fname + ".pkl")
        with open(dir_file, "rb") as fh:
            data = pickle.load(fh)
        with open(fname_gt, "rb") as fh:
            data_gt = pickle.load(fh)
            cols_gt = data_gt['cols']
            rows_gt = data_gt['rows']
        width = len(data['cols'])
        height = len(data['rows'])
        img = load_image(os.path.join(img_dir, fname), size=(width, height))

        cols = data['cols']
        # cols = np.where(data['cols'] > 0.5, 1, 0)
        rows = data['rows']
        # rows = np.where(data['rows'] > 0.5, 1, 0)
        if SKEW:
            cols = (cols, data['cols_start'])
            rows = (rows, data['rows_start'])
            cols_gt = (cols_gt, data_gt['cols_start'])
            rows_gt = (rows_gt, data_gt['rows_start'])

        cols = create_img(cols, height)
        rows = create_img(rows, width, reverse=True)
        img_tabled = img + (cols + rows)
        img_tabled = np.clip(img_tabled, 0, 1)

        cols_gt = create_img(cols_gt, height)
        rows_gt = create_img(rows_gt, width, reverse=True)
        img_tabled_gt = img + (cols_gt + rows_gt)
        img_tabled_gt = np.clip(img_tabled_gt, 0, 1)

        # show_cells(img)
        # show_cells(img)
        path_to_save = os.path.join(dir_to_save, fname + ".eps")

        show_img(img_tabled, img, img_tabled_gt, dest=path_to_save)
        print("Saved {}".format(path_to_save))



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "-h":
        main()
    else:
        print("Usage: python {} <dir with hyp pickle dir> <dir with real img> <dir with gt pkl> <dir_to_save>".format(sys.argv[0]))