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
First, the ROIs of the liver and tumors in CT images across four phases are segmented either manually or automatically. The original CT image and segmentation mask are placed in the **data** folder in the following form and name:

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
