# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 01:20:03 2019

@author: Think
"""

import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models

np.random.seed(0)


# load data 
INPUT_SIZE = 224
NUM_CLASSES = 20
data_dir = 'kaggle/'
labels = pd.read_csv(data_dir+'labels.csv')
print('Total number of training images: ',len(listdir(data_dir+'train')),'/ Labels(Should match num of images): ', len(labels))

selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
print('Selected dog breeds: ',selected_breed_list)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)


#a = labels_pivot.iloc[0, :].as_matrix().astype('float')
breeds = pd.DataFrame(labels_pivot.iloc[0, 1:]).index

# =============================================================================
# for i in breeds:
#     print(i)
# 
# =============================================================================
# train-test-split 80-20
train = labels_pivot.sample(frac=0.8)
valid = labels_pivot[~labels_pivot['id'].isin(train['id'])]
#print(train.shape, valid.shape)
#print(selected_breed_list)
#print(labels)



train.iloc[:,1:].idxmax(axis=1).value_counts().plot(kind='bar',color='yellowgreen',edges=False)

# =============================================================================
# 
# # overwrite the Dataset in pytorch
# class DogsDataset(Dataset):
#     def __init__(self, labels, root_dir, subset=False, transform=None):
#         self.labels = labels
#         self.root_dir = root_dir
#         self.transform = transform
#     
#     def __len__(self):
#         return len(self.labels)
#     
#     def __getitem__(self, idx):
#         img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
#         fullname = join(self.root_dir, img_name)
#         image = Image.open(fullname)
#         Labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')
#         Labels = np.argmax(Labels)
#         if self.transform:
#             image = self.transform(image)
#         return [image, Labels]
# 
# # z-socre normalization as required by ResNet50
# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225])
# 
# ds_trans = transforms.Compose(
#     [transforms.Resize(224),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      normalize])
# 
# train_ds = DogsDataset(train, data_dir+'train/', transform=ds_trans)
# valid_ds = DogsDataset(valid, data_dir+'train/', transform=ds_trans)
# 
# 
# train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
# valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=True, num_workers=0)
# 
# '''
# train_ds dimension ((3,224,224), label)
# '''
# # denormalization and show image
# def imshow(axis, inp):
#     """Denormalize and show"""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     axis.imshow((inp*255).astype(np.uint8))
#     
# img, label = next(iter(train_dl))
# #print(img.size(), label.size())
# 
# 
# # =============================================================================
# # fig = plt.figure(1, figsize=(16, 4))
# # grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.05)    
# # for i in range(img.size()[0]):
# #     ax = grid[i]
# #     imshow(ax, img[i])
# # 
# # =============================================================================
# 
# # load model for training
# use_gpu = torch.cuda.is_available()
# resnet = models.resnet50(pretrained=True)
# inputs, labels = next(iter(train_dl))
# if use_gpu:
#     resnet = resnet.cuda()
#     inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())   
# else:
#     inputs, labels = Variable(inputs), Variable(labels)
# outputs = resnet(inputs)
# outputs.size()
# 
# 
# # structure model
# def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#     use_gpu = torch.cuda.is_available()
#     best_model_wts = model.state_dict()
#     best_acc = -1.0
#     dataset_sizes = {'train': len(dataloders['train'].dataset), 
#                      'valid': len(dataloders['valid'].dataset)}
# 
#     for epoch in range(num_epochs):
#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train(True)
#             else:
#                 model.train(False)
# 
#             running_loss = 0.0
#             running_corrects = 0
#             
#             for inputs, labels in dataloders[phase]:
#                 if use_gpu:
#                     inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#                 else:
#                     inputs, labels = Variable(inputs), Variable(labels)
# 
#                 optimizer.zero_grad()
# 
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
# 
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
# 
#                 running_loss += loss.data
#                 running_corrects += torch.sum(preds == labels.data)
#             
#             if phase == 'train':
#                 train_epoch_loss = running_loss / dataset_sizes[phase]
#                 train_epoch_acc = running_corrects.item() / dataset_sizes[phase]
#             else:
#                 valid_epoch_loss = running_loss / dataset_sizes[phase]
#                 valid_epoch_acc = running_corrects.item() / dataset_sizes[phase]
#                 
#             if phase == 'valid' and valid_epoch_acc > best_acc:
#                 best_acc = valid_epoch_acc
#                 best_model_wts = model.state_dict()
# 
#         print('Epoch [{}/{}] train loss: {:.10f} acc: {:.10f} ' 
#               'valid loss: {:.10f} acc: {:.10f}'.format(
#                 epoch, num_epochs - 1,
#                 train_epoch_loss, train_epoch_acc, 
#                 valid_epoch_loss, valid_epoch_acc))
#             
#     print('Best val Acc: {:4f}'.format(best_acc))
#     print(running_corrects)
#     model.load_state_dict(best_model_wts)
#     return model
# 
# resnet = models.resnet50(pretrained=True)
# # freeze all model parameters
# for param in resnet.parameters():
#     param.requires_grad = False
# 
# # new final layer with 16 classes
# num_ftrs = resnet.fc.in_features
# resnet.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
# if use_gpu:
#     resnet = resnet.cuda()
# 
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# 
# dloaders = {'train':train_dl, 'valid':valid_dl}
# 
# '''
# we train the model on google codelabs already
# '''
# # =============================================================================
# # start_time = time.time()
# # model = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=3)
# # print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
# # =============================================================================
# 
# #torch.save(model, 'trained_model.pt')
# model = torch.load(data_dir+'trained_model.pt')
# 
# 
# def visualize_model(dataloders, model, num_images=25):
#     cnt = 0
#     fig = plt.figure(1, figsize=(16, 16))
#     grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.05)
#     for i, (inputs, labels) in enumerate(dataloders['valid']):
#         if use_gpu:
#             inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#         else:
#             inputs, labels = Variable(inputs), Variable(labels)
# 
#         outputs = model(inputs)
#         _, preds = torch.max(outputs.data, 1)
#         for j in range(inputs.size()[0]):
#           if preds[j] == labels.data[j]:
#              continue
#           else:
#             ax = grid[cnt]
#             imshow(ax, inputs.cpu().data[j])
#             ax.text(10, 210, '{}/{}'.format(breeds[preds[j]], breeds[labels.data[j]]), 
#                     color='k', backgroundcolor='w', alpha=0.8)
#             cnt += 1
#             if cnt == num_images:
#                 return
#             
# visualize_model(dloaders, resnet)
# 
# y_test = []
# y_pred = []
# for i, (inputs, labels) in enumerate(dloaders['valid']):
#   outputs = model(inputs)
#   _, preds = torch.max(outputs.data, 1)
#   y_test.extend(labels.numpy())
#   y_pred.extend(preds.numpy())
#   
# 
# 
# 
# # prediction part
# import matplotlib.ticker as ticker
# 
# all_categories = breeds
# def confusion_matrix(predictions, n_categories, testing_labels):
#     # Go through a bunch of examples and record which are correctly guessed
#     # Keep track of correct guesses in a confusion matrix
#     confusion = np.zeros((n_categories, n_categories))
#     cnt = 0
#     accuracy = 0
#     predictions = np.array(predictions)
#     testing_labels = np.array(testing_labels)
# 
#     for i in range(len(predictions)):
#         cnt += 1
#         predict = predictions[i]
#         target = testing_labels[i]
#         confusion[target][predict] += 1   
#         if predict == target:
#             accuracy += 1
#             
#     np.savetxt("nn.csv", confusion, delimiter=",")
#     # Normalize by dividing every row by its sum
#     total = [0 for _ in range(n_categories)]
#     for i in range(n_categories):
#         # 每一行做归一化
#         total[i] = confusion[i].sum()
#         confusion[i] = confusion[i] / confusion[i].sum()
#     
#     # Set up plot
#     fig = plt.figure(figsize=(30,30))
#     ax = fig.add_subplot(111)
#     cmap = plt.cm.get_cmap('Greens')
#     cax = ax.matshow(confusion, cmap=cmap)
#     fig.colorbar(cax)
#     
#     # Set up axes
#     ax.set_xticklabels([''] + all_categories, rotation=90)
#     ax.set_yticklabels([''] + all_categories)
#     
#     # Force label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#     
#     ind_array = np.arange(n_categories)
#     x, y = np.meshgrid(ind_array, ind_array)
#     for x_val, y_val in zip(x.flatten(), y.flatten()):
#         c = confusion[y_val][x_val]
#         if c > 0:
#             concrete = c*total[y_val]
#         else:
#             concrete = 0
#             
#         #这里是绘制数字，可以对数字大小和颜色进行修
#         #plt.text(x_val, y_val, "%d / %d" % (concrete, total[y_val]), color='black', fontsize=12, va='center', ha='center')
#           
#     # sphinx_gallery_thumbnail_number = 2
#     print(total)
#     plt.savefig('Confusion Matrix')
#     plt.show()
# 
# confusion_matrix(y_pred, len(breeds), y_test)
# 
# 
# 
# =============================================================================
