## libraries initialization #

import os
import numpy as np
from datasets import custom_dataset
from copy import copy
from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils import data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from unet import UNet
import random
import glob
import cv2

# initialization
img_path_raw = []
data_path = './datasets'

# path for image and mask
data_defect_blow_hole = os.path.join(data_path,'blow_hole/img_raw/')
data_defect_blow_hole_label = os.path.join(data_path,'blow_hole/img_mask/')

dir_img_raw = data_defect_blow_hole
dir_img_mask = data_defect_blow_hole_label

# get path of image
for img_path_tem in glob.glob(dir_img_raw+"*.jpg"):
    img_path_raw.append(img_path_tem)

## split path to train, validate and test #
# get length of data
length_data = len(img_path_raw)
# get length of train, validate and test
length_train = np.int(length_data * 0.7)
length_validate = np.int(length_data * 0.2)
length_test = np.int(length_data * 0.1)
# create random index
rand_ind = random.sample(range(0, length_data), length_data)

# get path for train, validate and test
img_path_train = [img_path_raw[i] for i in rand_ind[:length_train]]
img_path_validate = [img_path_raw[i] for i in rand_ind[length_train:length_train+length_validate]]
img_path_test = [img_path_raw[i] for i in rand_ind[length_train+length_validate:]]

## initialize dataset and dataloader #

# train
defects_dataset_train = custom_dataset(img_path_train,dir_img_mask)

defects_dataloader_train = DataLoader(defects_dataset_train, batch_size=4, shuffle=False)

# validate
defects_dataset_validate = custom_dataset(img_path_validate,dir_img_mask)

defects_dataloader_validate = DataLoader(defects_dataset_validate,batch_size=4, shuffle=False)

# test
defects_dataset_test = custom_dataset(img_path_test,dir_img_mask)

defects_dataloader_test = DataLoader(defects_dataset_test,batch_size=4, shuffle=False)

## visualize some sample images #

for i, img in enumerate(defects_dataloader_train):
    img_batch = img
    #break

n = 0
for i in range(n):
    # raw image is  img_batch[0][i].permute(1, 2, 0) and mask is img_batch[1][i][0]
    #im1 = cv2.cvtColor(np.array(img_batch[1][i][0]), cv2.COLOR_RGB2BGR)
    im1 = img_batch[0][i][0]
    im2 = img_batch[1][i][0]
    hmerge = np.hstack((im1, im2))  # 水平拼接
    cv2.imshow('name',hmerge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## get device #

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")

print(device)

## training parameters initialization #

model = UNet(in_channels=3, out_channels=1, init_features=32)
#https://github.com/mateuszbuda/brain-segmentation-pytorch

model = model.to(device)

# epoch number
num_epochs = 100

# initial learning rate
initial_lr = 0.001

# optimizer
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# loss function
loss_criterion = nn.BCELoss()

# scheduler for decaying learning rate
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1-(1/num_epochs))

## evaluation metric #
def numeric_score(prediction, groundtruth):
    prediction[prediction > 0.9] = 1;
    prediction[prediction <= 0.9] = 0;

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN


def accuracy(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0


## training and validation sets iterations #

# initialization
best_loss = 999
c_loss = 0
p_loss = 0
v_p_count = 0
v_p = 5
plot_epoch = np.arange(0, num_epochs)
plot_train_loss = np.zeros((num_epochs))
plot_validate_loss = np.zeros((num_epochs))
plot_validate_accuracy = np.zeros((num_epochs))
epoch_count = 0

# loop in epochs
barcode = tqdm(range(0, num_epochs))
for epoch in barcode:
    descript = 'epoch / num_epochs: %d / %d' % (epoch, num_epochs)
    barcode.set_description(desc=descript)
    # reduce learning rate usng scheduler
    scheduler.step()

    # get learning rate and print it
    lr = scheduler.get_lr()[0]

    print(' ')
    print('learn rate = ')
    print(lr)
    print(' ')

    # preallocate
    train_loss_total = 0.0
    train_accuracy_total = 0.0
    num_steps = 0

    # training loops
    #tbar = tqdm(defects_dataloader_train)
    for i, batch in enumerate(defects_dataloader_train):
        # set model to train mode
        model.train()
        # get train and label from dataloader
        batch_train = batch[0]

        batch_label = batch[1]
        batch_label[batch_label > 0] = 1

        # convert to cuda
        batch_train_in = batch_train.to(device, dtype=torch.float)
        batch_label_in = batch_label.to(device, dtype=torch.float)

        # Clean existing gradients
        optimizer.zero_grad()

        # forward pass of model
        predict_out = model(batch_train_in)

        # compute loss
        loss = loss_criterion(predict_out, batch_label_in)

        # add to total loss
        train_loss_total += loss.item()

        # backpropagation of gradients
        loss.backward()

        # update the parameters in optimizer
        optimizer.step()

        # get training accuracy
        train_accuracy_batch = accuracy(predict_out.cpu().detach().numpy(),
                                        batch_label_in.cpu().detach().numpy())

        # increase training accuracy
        train_accuracy_total += train_accuracy_batch

        # increase iteration count
        num_steps += 1

    # get training loss
    train_loss = train_loss_total / num_steps

    # get training accuracy
    train_accuracy = train_accuracy_total / num_steps

    # get values for plotting
    plot_train_loss[epoch] = train_loss

    ## validation step #########################################################

    # set model to evaluation mode
    model.eval()

    # preallocate
    validate_loss_total = 0.0
    validate_accuracy_total = 0.0
    num_steps = 0

    for i, batch in enumerate(defects_dataloader_validate):
        # get validate and label from dataloader
        batch_validate = batch[0]

        batch_validate_label = batch[1]
        batch_validate_label[batch_validate_label > 0] = 1

        # no grad in validation
        with torch.no_grad():
            # convert to cuda
            batch_validate_in = batch_validate.to(device, dtype=torch.float)
            batch_validate_label_in = batch_validate_label.to(device, dtype=torch.float)

            # get forward pass
            predict_out_validate = model(batch_validate_in)

            # compute loss
            loss = loss_criterion(predict_out_validate, batch_validate_label_in)

            # get total validation loss
            validate_loss_total += loss.item()

            # get validation accuracy
            validate_accuracy_batch = accuracy(predict_out_validate.cpu().detach().numpy(),
                                               batch_validate_label.cpu().detach().numpy())

            # increase validation accuracy
            validate_accuracy_total += validate_accuracy_batch

            # increase number of step
            num_steps += 1

    # get total validation loss
    validate_loss = validate_loss_total / num_steps

    # get validation accuracy
    validate_accuracy = validate_accuracy_total / num_steps

    # get values for plotting
    plot_validate_loss[epoch] = validate_loss
    plot_validate_accuracy[epoch] = validate_accuracy

    # increase epoch counter for reference purpose
    epoch_count += 1

    # update current and previous loss
    p_loss = copy(c_loss)
    c_loss = copy(validate_loss)

    # if current loss < best loss, save the model
    if c_loss < best_loss:
        v_p_count = 0
        best_loss = copy(validate_loss)
        print('  ')
        print('Saving low loss model ........................')
        print('  ')
        torch.save(model, 'model_best_loss_blow_hole.pt')

    # if current loss > previous loss, increase patience counter
    if c_loss > p_loss:
        v_p_count += 1

    # if validation patient reached, break from training
    if v_p_count > v_p:
        print('  ')
        print('Validate patience reached, break from training')
        print('  ')
        break

    print('\nTrain loss: {:.4f}, Training Accuracy: {:.4f} '.format(train_loss, train_accuracy))
    print('Val Loss: {:.4f}, Validation Accuracy: {:.4f} '.format(validate_loss, validate_accuracy))

## plot the train and validation loss + accuracy #

palette = plt.get_cmap('Set3')
plt.style.use('seaborn-darkgrid')


f1 = plt.figure(1)
ax1 =plt.plot(plot_epoch[0:epoch_count],plot_train_loss[0:epoch_count], linestyle = '--',marker='', color=palette(3), linewidth=3, alpha=0.9)
ax1 =plt.plot(plot_epoch[0:epoch_count],plot_validate_loss[0:epoch_count], linestyle = '-',marker='', color=palette(4), linewidth=3, alpha=0.9)
ax1 = plt.xlabel('epoch number')
ax1 = plt.ylabel('loss')
ax1 = plt.gca().legend(('training loss','validation loss'),loc='upper right')
ax1 = plt.title('losses against epoch number')

f2 = plt.figure(2)
ax2 =plt.plot(plot_epoch[0:epoch_count],plot_validate_accuracy[0:epoch_count], linestyle = '-',marker='', color=palette(2), linewidth=3, alpha=0.9)
ax2 = plt.xlabel('epoch number')
ax2 = plt.ylabel('accuracy')
ax2 = plt.gca().legend(('accuracy'),loc='lower right')
ax2 = plt.title('validation accuracy against epoch number')

## save the model #

torch.save(model, data_path+'model_trained_blow_hole.pt')

## testing set iterations #

plot_test_img = [None] * (3)
plot_test_mask = [None] * (3)
plot_test_predict = [None] * (3)

# set model to evaluation mode
model.eval()

# preallocate
test_accuracy_total = 0.0
test_loss_total = 0.0
num_steps = 0

for i, batch in enumerate(defects_dataloader_test):
    # get test and label from dataloader
    batch_test = batch[0]

    batch_test_label = batch[1]
    batch_test_label[batch_test_label > 0] = 1

    plot_test_img[i] = batch_test
    plot_test_mask[i] = batch_test_label

    # no grad in test
    with torch.no_grad():
        # convert to cuda
        batch_test_in = batch_test.to(device, dtype=torch.float)
        batch_test_label_in = batch_test_label.to(device, dtype=torch.float)

        # get forward pass
        predict_out_test = model(batch_test_in)
        plot_test_predict[i] = predict_out_test

        # compute loss
        loss = loss_criterion(predict_out_test, batch_test_label_in)

        # get total test loss
        test_loss_total += loss.item()

        # get test accuracy
        test_accuracy_batch = accuracy(predict_out_test.cpu().detach().numpy(),
                                       batch_test_label.cpu().detach().numpy())

        # increase test accuracy
        test_accuracy_total += test_accuracy_batch

        # increase number of step
        num_steps += 1

    # get total validation loss
    test_loss = test_loss_total / num_steps

    # get validation accuracy
    test_accuracy = test_accuracy_total / num_steps

    print('\nTest loss: {:.4f}, Test Accuracy: {:.4f} '.format(test_loss, test_accuracy))

## plot test results #
np.save(plot_test_img, 'plot_test_img.txt')

plt.style.use('default')
n = 0
k = 1
for i in range(len(plot_test_img)):

    for j in range(plot_test_img[i].shape[0]):

        img_tem_ori = plot_test_img[i][j]
        img_tem_mask = plot_test_mask[i][j]
        img_tem_predict = plot_test_predict[i][j]

        plot_ori = np.uint8(img_tem_ori.permute(1, 2, 0))
        plot_mask = img_tem_mask[0]
        plot_predict = img_tem_predict[0].cpu().detach().numpy()

        plt.figure(n)
        plt.title('input image '+str(k))
        plt.imshow(plot_ori)
        n +=1
        plt.figure(n)
        plt.title('input mask '+str(k))
        plt.imshow(plot_mask)
        n +=1
        plt.figure(n)
        plt.title('predicted output '+str(k))
        plt.imshow(plot_predict)
        n +=1
        k+=1
