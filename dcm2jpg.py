import glob
import os
import pydicom
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

dir = '/8TB_hdd/rsna-pneumonia-detection-challenge/stage_2_train_images/'

all_images = glob.glob(dir + '/**/*.dcm', recursive=True)

for dcm in tqdm(all_images):
    dicom = pydicom.dcmread(dcm)
    img = dicom.pixel_array
    monochrome = dicom.PhotometricInterpretation

    # ds.SetFileName(dcm)
    # img = sitk.GetArrayFromImage(sitk.ReadImage(dcm))[0]
    # img = np.asarray(img)

    img = cv2.resize(img, dsize=(320, 320))
    img = img - img.min()
    img = img / img.max()

    if monochrome == 'MONOCHROME2':
        pass
    elif monochrome == 'MONOCHROME1':
        img = 1. - img
    else:
        raise AssertionError()

    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    cv2.imwrite(dcm.replace('.dcm', '.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])