from PIL import Image
import cv2
import os
from tqdm import tqdm
import numpy as np
import multiprocessing
from functools import partial

def HiNGe(img):
    img = np.where(img > 200, 200, img)
    img_max = img.max()
    img_min = img.min()
    nor_img = (img - img_min) / (img_max - img_min)
    gamma = nor_img ** 1.5
    return gamma

def process(rawfile, rawpath, savedatapath):
    # 读取图像，保持深度
    img = cv2.imread(os.path.join(rawpath, rawfile), cv2.IMREAD_ANYDEPTH)
    
    if img is not None:
        # 将像素值为32767的像素点替换为0
        out = HiNGe(img)
        
        # 将 NumPy 数组转换为 PIL 图像对象
        out_pil = Image.fromarray(out)
        
        # 构建保存路径
        save_path = os.path.join(savedatapath, rawfile)
        
        # 保存为tiff文件
        out_pil.save(save_path)  # 将PIL图像对象保存到指定路径
        print(f"Processed and saved {rawfile}")

if __name__ == '__main__':
    rawpath = r'Z:\Yin_Hsu\yahan\Lectin'
    savedatapath = r'D:\yahan\lectin4'
    
    partial_func = partial(process, rawpath=rawpath, savedatapath=savedatapath)
    rawfilelist = os.listdir(rawpath)
    
    pool = multiprocessing.Pool(processes=10)
    list(tqdm(pool.imap(partial_func, rawfilelist), total=len(rawfilelist)))
