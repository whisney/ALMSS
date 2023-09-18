# ALMSS
This is the repository for the paper "Automatic and noninvasive origin diagnosis of liver metastases via hierarchical artificial-intelligence system trained on multiphasic CT data".

![ALMSS](/pic/ALMSS.png)

## Requirements
* python 3.6
* pytorch 1.5+
* scikit-learn
* scikit-image
* tensorboardX
* SimpleITK

## ROIs segmentation
First, the ROIs of the liver and tumors in CT images across four phases are segmented either manually or automatically. The original CT images and segmentation masks should be placed in the **data** folder in the following form and name:
```
- data
  - ID001
    - Plain_img.nii.gz
    - Plain_liver_mask.nii.gz
    - Plain_Tumor_mask.nii.gz
    - Arterial_img.nii.gz
    - Arterial_liver_mask.nii.gz
    - Arterial_Tumor_mask.nii.gz
    - Venous_img.nii.gz
    - Venous_liver_mask.nii.gz
    - Venous_Tumor_mask.nii.gz
    - Delay_img.nii.gz
    - Delay_liver_mask.nii.gz
    - Delay_Tumor_mask.nii.gz
  - ID002
    - ...
  - ...
- README.md
- LICENSE
- ...
```
We used [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) to train the models for automatic segmentation of liver and tumors. Specifically, we used the entire abdominal CT image as input for the liver segmentation model. In contrast, for the liver tumor segmentation model, we employed only the segmented ROI of the liver as input. Both models were trained in a 3D full-resolution mode.

## Data preprocessing
Before starting training, ROI region needs to be cropped and normalized. You can run this step with the following command:
```
cd ALMSS
python preprocessing.py
```
An Excel file, which stores information on gender, age, and labels, is required and is presented in the format below:

| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |
