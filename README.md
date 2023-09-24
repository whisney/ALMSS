# ALMSS
This is the repository for the paper "Automatic and noninvasive origin diagnosis of liver metastases via hierarchical artificial-intelligence system trained on multiphasic CT data".

![ALMSS](/pic/ALMSS.png)

## Requirements
* python 3.6
* pytorch 1.5+
* pandas
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
An Excel file (metadata.xlsx), which stores information on gender, age, and labels, is required and is presented in the format below:

| ID | age | sex | label |
| :-----: | :----: | :----: | :----: |
| ID001 | 53 | male | 1 |
| ID002 | 51 | female | 2 |
| ID003 | 39 | male | 5 |
| ... | ... | ... | ... |

where the labels are the tumor source (1-Intestine; 2-Lung; 3-Breast; 4-Oesophagogastric; 5-Pancreatobiliary; 6-Reproductive; 7-HCC; 8-ICC).
All data is split into training, validation, and test sets. The respective patient IDs are saved in the .txt file for each subset. Please place the Excel and .txt files in the **relevant_files** folder：
```
- relevant_files
  - metadata.xlsx
  - train.txt
  - val.txt
  - test.txt
- data
- README.md
- ...
```

## Model training
For training primary/metastatic tumors classification model, you can run:
```
python train_primary.py --gpu 0 --bs 10 --epoch 200 --seed 42
```
For training classification model of metastatic tumor source, you can run:
```
python train_metastatic.py --gpu 0 --bs 10 --epoch 200 --seed 42
```

## Model evaluation
To evaluate the two models, you can run:
```
# for primary/metastatic tumors classification model
python evaluate_primary.py --gpu 0 --model_path trained_models/primary_tumor/bs10_epoch200_seed42/best_AUC_val.pth --set test

# for classification model of metastatic tumor source
python evaluate_metastatic.py --gpu 0 --model_path trained_models/metastatic_tumor/bs10_epoch200_seed42/best_AUC_val.pth --set test
```
