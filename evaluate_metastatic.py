from ResNet_3D import ResNet18_3D_4stream_clinical_LSTM
import os
import argparse
import shutil
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from Nii_utils import NiiDataRead
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--model_path', type=str, help='trained model root')
parser.add_argument('--set', type=str, help='trained model root')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

data_dir = 'data_preprocessed'
metadata_path = 'relevant_files/metadata.xlsx'
split_path = 'relevant_files/{}.txt'.format(args.set)

metadata_df = pd.read_excel(metadata_path)
metadata_df['ID'] = metadata_df['ID'].astype(str)
metadata_df['label'] = metadata_df['label'].astype(int)

with open(split_path, 'r') as f:
    ID_list_orginal = f.readlines()
ID_list_orginal = [n.strip('\n') for n in ID_list_orginal]

new_dir = args.model_path.rstrip('.pth') + '_{}'.format(args.set)
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
print(new_dir)
os.makedirs(new_dir, exist_ok=True)

net = ResNet18_3D_4stream_clinical_LSTM(in_channels=2, clinical_inchannels=3, n_classes=6, pretrained=False, no_cuda=False).cuda()

net.load_state_dict(torch.load(args.model_path))
net.eval()

with torch.no_grad():
    ID_list = []
    gender_list = []
    age_list = []
    label_list = []

    for ID in ID_list_orginal:
        label_one = metadata_df.loc[metadata_df.ID == ID, 'label'].values[0]
        if label_one <= 6:
            ID_list.append(ID)
            label_list.append(label_one - 1)
            gender_list.append(str(metadata_df.loc[metadata_df.ID == ID, 'sex'].values[0]))
            age_list.append(int(metadata_df.loc[metadata_df.ID == ID, 'age'].values[0]))

    one_hot_label_all = []
    pred_scores_all = []
    class_label_all = []
    pred_class_all = []

    for i, ID in enumerate(ID_list):
        gender = gender_list[i]
        age = age_list[i]
        label = label_list[i]
        labels_one_hot = torch.tensor(label).long()
        labels_one_hot = torch.zeros((1, 6)).scatter_(1, labels_one_hot.unsqueeze(0).unsqueeze(0), 1).float()

        if 'male' == gender:
            gender = torch.from_numpy(np.array([1, 0]))
        elif 'female' == gender:
            gender = torch.from_numpy(np.array([0, 1]))

        age = torch.from_numpy(np.array([age])).float() / 100

        Plain_img_liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Plain_img_liver.nii.gz'))
        Arterial_img_liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Arterial_img_liver.nii.gz'))
        Venous_img_liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Venous_img_liver.nii.gz'))
        Delay_img_liver, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Delay_img_liver.nii.gz'))

        Plain_img_tumor, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Plain_img_tumor.nii.gz'))
        Arterial_img_tumor, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Arterial_img_tumor.nii.gz'))
        Venous_img_tumor, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Venous_img_tumor.nii.gz'))
        Delay_img_tumor, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Delay_img_tumor.nii.gz'))

        Plain_img = torch.from_numpy(np.concatenate((Plain_img_liver[np.newaxis, ...], Plain_img_tumor[np.newaxis, ...]), axis=0)).unsqueeze(0).float().cuda()
        Arterial_img = torch.from_numpy(np.concatenate((Arterial_img_liver[np.newaxis, ...], Arterial_img_tumor[np.newaxis, ...]), axis=0)).unsqueeze(0).float().cuda()
        Venous_img = torch.from_numpy(np.concatenate((Venous_img_liver[np.newaxis, ...], Venous_img_tumor[np.newaxis, ...]), axis=0)).unsqueeze(0).float().cuda()
        Delay_img = torch.from_numpy(np.concatenate((Delay_img_liver[np.newaxis, ...], Delay_img_tumor[np.newaxis, ...]), axis=0)).unsqueeze(0).float().cuda()

        gender_age = torch.cat((gender, age)).unsqueeze(0).float().cuda()

        output = net(Plain_img, Arterial_img, Venous_img, Delay_img, gender_age)
        output = torch.softmax(output, dim=1)

        predicted = torch.argmax(output, dim=1, keepdim=False).detach()
        pred_scores_all.append(output.detach().cpu())
        one_hot_label_all.append(labels_one_hot)
        class_label_all.append(label.cpu().numpy())
        pred_class_all.append(predicted.cpu().numpy())

        print("{}/{}  ID: {}  Label: {}  Pred_score: {}".format(i+1, len(ID_list), ID, label, output.squeeze(0).detach().cpu().numpy()))

    one_hot_label_all = torch.cat(one_hot_label_all, dim=0).numpy().astype(np.uint8)
    pred_scores_all = torch.cat(pred_scores_all, dim=0).numpy()
    class_label_all = np.concatenate(class_label_all)
    pred_class_all = np.concatenate(pred_class_all)

    AUC = roc_auc_score(one_hot_label_all, pred_scores_all, multi_class='ovr')
    ACC = accuracy_score(class_label_all, pred_class_all)
    print('Set: {}  ACC: {}  AUC:  {}'.format(args.set, ACC, AUC))

    np.save(os.path.join(new_dir, 'label_one_hot.npy'), one_hot_label_all)
    np.save(os.path.join(new_dir, 'score.npy'), pred_scores_all)
