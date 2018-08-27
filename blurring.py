from multiprocessing import Pool
import cv2
import glob
import numpy as np

def blur(img_path):
    img = cv2.imread(img_path)

    blurred = cv2.GaussianBlur(img, (7,7), 5)

    img_name = img_path.split('/')[-1]

    cv2.imwrite('data/sah_blur/{}'.format(img_name), blurred)

if __name__ == '__main__':
    
    pool = Pool(4)
    paths = glob.glob("data/sah/*")

    pool.map(blur,paths)

