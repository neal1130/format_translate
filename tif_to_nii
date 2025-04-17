from PIL import Image
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import multiprocessing as mp

def convert_tif_to_nii(args):
    tif_file, rawfile, output_nii_folder,idx = args
    with Image.open(os.path.join(tif_file, rawfile)) as img:
        tif_data = np.asarray(img)
        # 交换XY轴
        tif_data = np.transpose(tif_data, (1, 0))
    tif_data=np.where(tif_data>0,1,0).astype(np.uint8)    # mask 用
    # Create NIfTI image object
    nii_img = nib.Nifti1Image(tif_data, np.eye(4))

    # 使用原始文件名
    #output_filename = os.path.join(output_nii_folder, f"{os.path.splitext(rawfile)[0]}_0000.nii.gz")
    #output_filename = os.path.join(output_nii_folder, f"{os.path.splitext(rawfile)[0]}.nii.gz") #mask
    output_filename = os.path.join(output_nii_folder, f"{idx}.nii.gz")

    
    
    nib.save(nii_img, output_filename)

def tif_to_nii(tif_folder, output_nii_folder):
    rawfilelist = sorted(os.listdir(tif_folder))
    file_infos = [(tif_folder, rawfile, output_nii_folder,idx+1) for idx ,rawfile in enumerate(rawfilelist)]

    max_processes = max(1, mp.cpu_count())  # 至少保证1个进程

    with mp.Pool(processes=max_processes) as pool:
        list(tqdm(pool.imap_unordered(convert_tif_to_nii, file_infos), total=len(file_infos)))


def convert_nii_to_tif(args):
    nii_file_path, output_tif_folder, rawfile = args
    
    # Load the NIfTI image
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()
    
    # Ensure data is 2D
    if nii_data.ndim != 2:
        print(f"Error: NIfTI image at {nii_file_path} is not 2D.")
        return

    # Normalize the image data to 0-255 and convert to uint8
    if nii_data.dtype != np.uint16:
        nii_data = np.where(nii_data > 0, 65535, 0).astype(np.uint16)
        
    nii_data_normalized = np.transpose(nii_data, (1, 0))
    # Save the 2D image as a TIFF file
    slice_img = Image.fromarray(nii_data_normalized)
    # 去掉文件名的所有擴展名並加上 .tif
    output_filename = os.path.join(output_tif_folder, f"{os.path.splitext(os.path.splitext(rawfile)[0])[0]}.tif")
    slice_img.save(output_filename)

def nii_to_tif(nii_folder, output_tif_folder):
    nii_filelist = [f for f in sorted(os.listdir(nii_folder)) if f.endswith('.nii.gz')]
    file_infos = [(os.path.join(nii_folder, nii_file), output_tif_folder, nii_file) 
                  for nii_file in nii_filelist]

    max_processes = max(1, mp.cpu_count())

    with mp.Pool(processes=max_processes) as pool:
        list(tqdm(pool.imap_unordered(convert_nii_to_tif, file_infos), total=len(file_infos)))

if __name__ == '__main__':
    print("cpu_count", mp.cpu_count())
   
    input_folder = r'D:\Lewan\raw_data\label'
    output_folder = r'D:\nnUNet\DataSet\nnUnet_raw\Dataset225_cfos2\labelsTr'
    
    # 轉換 TIFF 到 NIfTI
    tif_to_nii(input_folder,output_folder)

    # 轉換 NIfTI 到 TIFF
   # nii_to_tif(input_folder, output_folder)
