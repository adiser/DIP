from multiprocessing import Pool
import cv2
import glob
import numpy as np

def sharpen(img_path):
    img = cv2.imread(img_path)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)

    img_name = img_path.split('/')[-1]

    cv2.imwrite('data/sdh_sharp/{}'.format(img_name), sharpened)

if __name__ == '__main__':
    
    pool = Pool(4)
    paths = glob.glob("data/sdh/*")

    pool.map(sharpen,paths)

