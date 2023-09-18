import os
from Nii_utils import NiiDataRead, NiiDataWrite
import numpy as np
from skimage import transform

taget_size = (32, 160, 192)
values_clip = (-55, 145)
data_dir = r'data'
save_dir = r'data_preprocessed'

for ID in os.listdir(data_dir):
    print(ID)
    os.makedirs(os.path.join(save_dir, ID), exist_ok=True)
    for mode in ['Plain', 'Arterial', 'Venous', 'Delay']:
        img, spacing, origin, direction = NiiDataRead(os.path.join(data_dir, ID, '{}_img.nii.gz'.format(mode)))
        mask_liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, '{}_liver_mask.nii.gz'.format(mode)))
        mask_tumor, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, '{}_Tumor_mask.nii.gz'.format(mode)))

        z_, x_, y_ = mask_liver.nonzero()
        z1 = z_.min()
        z2 = z_.max()
        x1 = x_.min()
        x2 = x_.max()
        y1 = y_.min()
        y2 = y_.max()

        img = img[z1: z2 + 1, x1: x2 + 1, y1:y2 + 1]
        mask_liver = mask_liver[z1: z2 + 1, x1: x2 + 1, y1:y2 + 1]
        mask_tumor = mask_tumor[z1: z2 + 1, x1: x2 + 1, y1:y2 + 1]

        img = np.clip(img, values_clip[0], values_clip[1])
        img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

        spacing_z = (spacing[0] * img.shape[0]) / taget_size[0]
        spacing_x = (spacing[1] * img.shape[1]) / taget_size[1]
        spacing_y = (spacing[2] * img.shape[2]) / taget_size[2]

        img = transform.resize(img, taget_size, order=0, mode='constant',
                               clip=False, preserve_range=True, anti_aliasing=False)
        mask_liver = transform.resize(mask_liver, taget_size, order=0, mode='constant',
                                     clip=False, preserve_range=True, anti_aliasing=False)
        mask_tumor = transform.resize(mask_tumor, taget_size, order=0, mode='constant',
                                      clip=False, preserve_range=True, anti_aliasing=False)

        img_liver = np.copy(img)
        img_tumor = np.copy(img)
        img_liver[mask_liver == 0] = -1
        img_tumor[mask_tumor == 0] = -1

        NiiDataWrite(os.path.join(save_dir, ID, '{}_img_liver.nii.gz'.format(mode)), img_liver,
                     np.array([spacing_z, spacing_x, spacing_y]), origin, direction)
        NiiDataWrite(os.path.join(save_dir, ID, '{}_img_tumor.nii.gz'.format(mode)), img_tumor,
                     np.array([spacing_z, spacing_x, spacing_y]), origin, direction)