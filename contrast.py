from multiprocessing import Pool
import cv2
import glob
import numpy as np
from torchvision.transforms.functional import adjust_contrast
from PIL import Image

def contrast(img_path):
    img = Image.open(img_path)

    res = adjust_contrast(img, 2)

    img_name = img_path.split('/')[-1]

    res.save('data/sdh_contrast/{}'.format(img_name))

if __name__ == '__main__':
    
    pool = Pool(4)
    paths = glob.glob("data/sdh/*")

    pool.map(contrast,paths)

