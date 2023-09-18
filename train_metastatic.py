import os
from dataset import Dataset_for_metastatic_tumor
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
from ResNet_3D import ResNet18_3D_4stream_clinical_LSTM
from loss_function.CB_Loss import CB_loss
import argparse
import random
import shutil
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

lr_max = 0.0002
L2 = 0.00005
input_size = (32, 160, 192)
data_dir = 'data_preprocessed'
metadata_path = 'relevant_files/metadata.xlsx'
train_split_path = 'relevant_files/train.txt'
val_split_path = 'relevant_files/val.txt'
num_class = 6

save_dir = 'trained_models/metastatic_tumor/bs{}_epoch{}_seed{}'.format(args.bs, args.epoch, args.seed)

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
print(save_dir)

print('dataset loading')

train_data = Dataset_for_metastatic_tumor(data_dir, train_split_path, metadata_path, augment=True)
val_data = Dataset_for_metastatic_tumor(data_dir, val_split_path, metadata_path, augment=False)


train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, drop_last=False)

print('train_lenth: %i  val_lenth: %i  num_0: %i  num_1: %i  num_2: %i  num_3: %i  num_4: %i  num_other: %i' % (
    train_data.len, val_data.len, train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3, train_data.num_4, train_data.num_5))

net = ResNet18_3D_4stream_clinical_LSTM(in_channels=2, clinical_inchannels=3, n_classes=num_class, pretrained=False, no_cuda=False).cuda()

optimizer = optim.AdamW(net.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((6 / 10) * args.epoch), int((9 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)

best_AUC_val = 0
best_ACC_val = 0

print('training')

for epoch in range(args.epoch):
    net.train()
    train_epoch_loss = []
    train_epoch_one_hot_label = []
    train_epoch_pred_scores = []
    train_epoch_class_label = []
    train_epoch_pred_class = []
    for i, (Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages, labels) in enumerate(train_dataloader):
        Plain_imgs = Plain_imgs.cuda().float()
        Arterial_imgs = Arterial_imgs.cuda().float()
        Venous_imgs = Venous_imgs.cuda().float()
        Delay_imgs = Delay_imgs.cuda().float()
        gender_ages, labels = gender_ages.cuda().float(), labels.cuda().long()
        labels_one_hot = torch.zeros((labels.size(0), num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
        optimizer.zero_grad()
        outputs = net(Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages)
        loss = CB_loss(labels, outputs, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3, train_data.num_4, train_data.num_5],
                       no_of_classes=num_class, loss_type='focal', beta=0.999, gamma=2)
        loss.backward()
        optimizer.step()
        outputs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
        train_epoch_pred_scores.append(outputs.detach().cpu())
        train_epoch_one_hot_label.append(labels_one_hot)
        train_epoch_loss.append(loss.item())
        train_epoch_class_label.append(labels.cpu().numpy())
        train_epoch_pred_class.append(predicted.cpu().numpy())
        print('[%d/%d, %d/%d] train_loss: %.3f' %
              (epoch + 1, args.epoch, i + 1, len(train_dataloader), loss.item()))
    lr_scheduler.step()

    with torch.no_grad():
        net.eval()
        val_epoch_loss = []
        val_epoch_label = []
        val_epoch_pred_scores = []
        val_epoch_class_label = []
        val_epoch_pred_class = []
        for i, (Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages, labels) in enumerate(val_dataloader):
            Plain_imgs = Plain_imgs.cuda().float()
            Arterial_imgs = Arterial_imgs.cuda().float()
            Venous_imgs = Venous_imgs.cuda().float()
            Delay_imgs = Delay_imgs.cuda().float()
            gender_ages, labels = gender_ages.cuda().float(), labels.cuda().long()
            labels_one_hot = torch.zeros((labels.size(0), num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
            outputs = net(Plain_imgs, Arterial_imgs, Venous_imgs, Delay_imgs, gender_ages)
            loss = CB_loss(labels, outputs,
                           samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2, train_data.num_3,
                                            train_data.num_4, train_data.num_5],
                           no_of_classes=num_class, loss_type='focal', beta=0.999, gamma=2)
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
            val_epoch_pred_scores.append(outputs.detach().cpu())
            val_epoch_label.append(labels_one_hot)
            val_epoch_loss.append(loss.item())
            val_epoch_class_label.append(labels.cpu().numpy())
            val_epoch_pred_class.append(predicted.cpu().numpy())

    train_epoch_one_hot_label = torch.cat(train_epoch_one_hot_label, dim=0).numpy().astype(np.uint8)
    train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy()
    val_epoch_label = torch.cat(val_epoch_label, dim=0).numpy().astype(np.uint8)
    val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()

    train_epoch_class_label = np.concatenate(train_epoch_class_label)
    train_epoch_pred_class = np.concatenate(train_epoch_pred_class)
    val_epoch_class_label = np.concatenate(val_epoch_class_label)
    val_epoch_pred_class = np.concatenate(val_epoch_pred_class)

    train_AUC = roc_auc_score(train_epoch_one_hot_label, train_epoch_pred_scores)
    val_AUC = roc_auc_score(val_epoch_label, val_epoch_pred_scores)

    train_ACC = accuracy_score(train_epoch_class_label, train_epoch_pred_class)
    val_ACC = accuracy_score(val_epoch_class_label, val_epoch_pred_class)

    train_epoch_loss = np.mean(train_epoch_loss)
    val_epoch_loss = np.mean(val_epoch_loss)

    print(
        '[%d/%d] train_loss: %.3f train_AUC: %.3f val_AUC: %.3f train_ACC: %.3f val_ACC: %.3f' %
        (epoch, args.epoch, train_epoch_loss, train_AUC, val_AUC, train_ACC, val_ACC))

    if val_AUC > best_AUC_val:
        best_AUC_val = val_AUC
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_AUC_val.pth'))
    if val_ACC > best_ACC_val:
        best_ACC_val = val_ACC
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_ACC_val.pth'))
    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

    train_writer.add_scalar('loss', train_epoch_loss, epoch)
    train_writer.add_scalar('AUC', train_AUC, epoch)
    train_writer.add_scalar('ACC', train_ACC, epoch)

    val_writer.add_scalar('loss', val_epoch_loss, epoch)
    val_writer.add_scalar('AUC', val_AUC, epoch)
    val_writer.add_scalar('ACC', val_ACC, epoch)
    val_writer.add_scalar('best_AUC_val', best_AUC_val, epoch)
    val_writer.add_scalar('best_ACC_val', best_ACC_val, epoch)

train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)