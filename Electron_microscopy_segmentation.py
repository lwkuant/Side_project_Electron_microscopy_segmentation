# -*- coding: utf-8 -*-
"""
Segment the interesting parts under the electron microscopy picture
"""

### Load required packages
import pandas as pd 
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from skimage.io import imread
from collections import Counter
from time import time 

### Load the data
train_img = imread('D:/Dataset/Side_project_Electron_microscopy_3d_segmentation/training.tif')
train_img_truth = imread('D:/Dataset/Side_project_Electron_microscopy_3d_segmentation/training_groundtruth.tif')

### show an example
pic = 100 
fig = plt.figure(figsize=(15, 4))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(train_img[pic], cmap='gray')
plt.grid(False)

ax_2 = fig.add_subplot(1, 3, 2)
plt.hist(train_img[pic].flatten())

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(train_img_truth[pic], cmap='gray')
plt.grid(False)

### show the segmentation 
fig = plt.figure(figsize=(12, 10))
ax_1 = fig.add_subplot(2, 2, 1)
plt.imshow(train_img[pic], cmap='gray')
plt.grid(False)
ax_2 = fig.add_subplot(2, 2, 2)
plt.hist(train_img[pic].flatten())


ax_3 = fig.add_subplot(2, 2, 3)
plt.imshow(train_img_truth[pic]*train_img[pic], cmap='gray')
plt.grid(False)
ax_4 = fig.add_subplot(2, 2, 4)
#plt.hist((train_img_truth[pic]*train_img[pic]).flatten())
plt.hist((train_img[pic][train_img_truth[pic]>0]))


### Using the threshold from histogram to segment
minimum = (train_img[pic]*[train_img_truth[pic]>0]).min()
maximum = (train_img[pic]*[train_img_truth[pic]>0]).max()
fig = plt.figure(figsize=(15, 4))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(train_img[pic], cmap='gray')
plt.grid(False)

ax_2 = fig.add_subplot(1, 3, 2)
plt.imshow((train_img[pic]>minimum)&(train_img[pic]<maximum), cmap='gray')
plt.grid(False)

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(train_img_truth[pic]>0, cmap='gray')
plt.grid(False)
# loks not good enough...

### Evaluate the sgementation (on the benchmark segmentation)
y = ((train_img_truth[pic]>0)*1).flatten()
y_pred = (1 - train_img[pic]/255).flatten()

from sklearn.metrics import roc_auc_score, roc_curve, auc

fpr, tpr, th = roc_curve(y, y_pred)
score = roc_auc_score(y, y_pred)
score

fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % score)
ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for Benchmark Segmentation')
ax.legend(loc="lower right")

### Evaluate the segmentation using hand-tuned histogram
y_pred = ((train_img[pic]>=minimum)&(train_img[pic]<=maximum))
y_pred = y_pred.flatten()

score_tab = pd.crosstab(y, y_pred)           
sns.heatmap(score_tab)
plt.ylabel('Actual')
plt.xlabel('Prediction')

print(score_tab)
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
# the precision is awful
plt.imshow(((train_img[pic]>=minimum)&(train_img[pic]<=maximum))*1)
plt.imshow(train_img_truth[pic])

### gaussian thresholding
from skimage.filters import gaussian
fig = plt.figure(figsize=(10, 4))
ax_1 = fig.add_subplot(1, 2, 1)
plt.imshow(gaussian(train_img[1], sigma=4))

ax_2 = fig.add_subplot(1, 2, 2)
plt.imshow(train_img[1])

### transform the images using gaussian thresholding
train_img = train_img/255

#from skimage.filters import gaussian
#for i in range(len(train_img)):
#    train_img[i] = gaussian(train_img[i], sigma=4)

### split the data into training and test dataset
seed = 100
np.random.seed(seed)
ind = np.random.choice(range(train_img.shape[0]), int(train_img.shape[0]*0.9),
                           replace=False)

pic_x = np.copy(train_img)
pic_y = np.copy(train_img_truth)/255

pic_x_tr = pic_x[ind]
pic_y_tr = pic_y[ind]
pic_x_test = pic_x[np.setdiff1d(range(train_img.shape[0]), ind)]
pic_y_test = pic_y[np.setdiff1d(range(train_img.shape[0]), ind)]

### reshape the data for cnn
pic_x_tr = pic_x_tr.reshape([len(pic_x_tr), 768, 1024, 1])
pic_y_tr = pic_y_tr.reshape([len(pic_y_tr), 768, 1024, 1])
pic_x_test = pic_x_test.reshape([len(pic_x_test), 768, 1024, 1])
pic_y_test = pic_y_test.reshape([len(pic_y_test), 768, 1024, 1])

### build the model structure
seed = 100
np.random.seed(seed)
import keras

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

height = 768
width = 1024

input_img = Input(shape=(768, 1024, 1))

####
#x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
#x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
#encoded = MaxPooling2D((2, 2), name='block1_pool')(x)

# Block 2
#x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
#encoded = MaxPooling2D((2, 2), name='block2_pool')(x)

## Block 3
#x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
#encoded = MaxPooling2D((2, 2), name='block3_pool')(x)

# Block 4
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#x = MaxPooling2D((2, 2), name='block4_pool')(x)

# Block 5
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#encoded = MaxPooling2D((2, 2), name='block5_pool')(x)


#x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoded)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#
#x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
#x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#
#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)

#x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
#x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#
#decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
#autoencoder = Model(input_img, decoded)
#
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',  metrics=['acc'])
#
#autoencoder.fit(pic_x_tr, pic_y_tr,
#                epochs=30,
#                batch_size=1,
#                shuffle=True,
#                validation_data=(pic_x_test, pic_y_test),
#                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

##

""" Original neural structure
input_img = Input(shape=(768, 1024, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this moment, the shape of the image is (8, 96, 128)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
"""

input_img = Input(shape=(768, 1024, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this moment, the shape of the image is (48, 64, 8)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# 20180421: Bad
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)
#
## at this moment, the shape of the image is (48, 64, 8)
#
#x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['acc']) # 2017/5/7
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc']) # 2018/4/20, worse thant adadelta
#autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['acc']) # 2018/4/21, worse thant adadelta

start_time = time()
autoencoder.fit(pic_x_tr, 
                pic_y_tr,
                epochs=50,
                batch_size=3,
                validation_data = [pic_x_test, pic_y_test],
                shuffle=True,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
print(time() - start_time)


y_tr_pred = autoencoder.predict(pic_x_tr, batch_size=3)
y_test_pred = autoencoder.predict(pic_x_test, batch_size=3)

pic = 0
fig = plt.figure(figsize=(18, 8))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(pic_x_tr[pic].reshape([768, 1024]), cmap='gray')
ax_1.set_title('Raw Picture', fontsize=15)
plt.grid(False)

ax_2 = fig.add_subplot(1, 3, 2)
plt.imshow(y_tr_pred[pic].reshape([768, 1024]), cmap='gray')
ax_2.set_title('Prediction from Simple CNN model', fontsize=15)
plt.grid(False)

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(pic_y_tr[pic].reshape([768, 1024]), cmap='gray')
ax_3.set_title('Ground Truth', fontsize=15)
plt.grid(False)


pic = 0
fig = plt.figure(figsize=(18, 8))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(pic_x_test[pic].reshape([768, 1024]), cmap='gray')
ax_1.set_title('Raw Picture', fontsize=15)
plt.grid(False)

ax_2 = fig.add_subplot(1, 3, 2)
plt.imshow(y_test_pred[pic].reshape([768, 1024])>0.5, cmap='gray')
ax_2.set_title('Prediction from Simple CNN model', fontsize=15)
plt.grid(False)

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(pic_y_test[pic].reshape([768, 1024]), cmap='gray')
ax_3.set_title('Ground Truth', fontsize=15)
plt.grid(False)

### use threshold
fig = plt.figure(figsize=[12, 8])
ax_1 = fig.add_subplot(1, 2, 1)
plt.imshow((y_test_pred[pic].reshape([768, 1024])>0.55)*pic_x_test[pic].reshape([768, 1024]),
            cmap='gray')
plt.grid(False)

ax_2 = fig.add_subplot(1, 2, 2)
plt.imshow(pic_y_test[pic].reshape([768, 1024]), cmap='gray')
plt.grid(False)

### Evaluate the model
from sklearn.metrics import log_loss
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## 1. Baseline
# Accuracy
print(np.mean([accuracy_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (1 - (pic_x_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]*1)) for i in range(pic_x_tr.shape[0])]))
# 0.671550931157

# Precision
print(np.mean([precision_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (1 - (pic_x_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]*1)) for i in range(pic_x_tr.shape[0])]))
# 0.129430356929

# Recall
print(np.mean([recall_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (1 - (pic_x_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]*1)) for i in range(pic_x_tr.shape[0])]))
# 0.864888246063

# F1 score
print(np.mean([f1_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (1 - (pic_x_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]*1)) for i in range(pic_x_tr.shape[0])]))
# 0.223435378402

# ROC-AUC
print(np.mean([roc_auc_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (1 - pic_x_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i])) for i in range(pic_x_tr.shape[0])]))
# 0.806822370082

## 2. Without tuning the threshold
threshold = 0.5

## Training
# Accuracy
print(np.mean([accuracy_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_tr_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_tr.shape[0])]))
# 0.985303672584

# Precision
print(np.mean([precision_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_tr_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_tr.shape[0])]))
# 0.909384893806

# Recall
print(np.mean([recall_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_tr_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_tr.shape[0])]))
# 0.823150863226

# F1 score
print(np.mean([f1_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_tr_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_tr.shape[0])]))
# 0.863054677176

# ROC-AUC
print(np.mean([roc_auc_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], y_tr_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i]) for i in range(pic_x_tr.shape[0])]))
# 0.996151093209

## Testing 
# Accuracy
print(np.mean([accuracy_score(pic_y_test.reshape([-1, np.prod(y_test_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_test_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.986047931746

# Precision
print(np.mean([precision_score(pic_y_test.reshape([-1, np.prod(y_test_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_test_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.899365174427

# Recall
print(np.mean([recall_score(pic_y_test.reshape([-1, np.prod(y_test_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_test_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.83250794642

# F1 score
print(np.mean([f1_score(pic_y_test.reshape([-1, np.prod(y_test_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_test_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.863340686965

# ROC-AUC
print(np.mean([roc_auc_score(pic_y_test.reshape([-1, np.prod(y_test_pred.shape[1:])])[i], y_test_pred.reshape([-1, np.prod(y_test_pred.shape[1:])])[i]) for i in range(pic_x_test.shape[0])]))
# 0.996394102766

## 3. Tuning the threshold

# Aimming on the "F1 score"
threshold_list = np.arange(0, 1, 0.05)

f1_score_list = [np.mean([f1_score(pic_y_tr.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_tr_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=t)[i]) for i in range(pic_x_tr.shape[0])])\
                         for t in threshold_list]

print(threshold_list[np.argmax(f1_score_list)]) # 0.35
print(f1_score_list[np.argmax(f1_score_list)]) # 0.8936087516718253

threshold = threshold_list[np.argmax(f1_score_list)]

# Accuracy
print(np.mean([accuracy_score(pic_y_test.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.98571695066

# Precision
print(np.mean([precision_score(pic_y_test.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.854271132129

# Recall
print(np.mean([recall_score(pic_y_test.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.882002959165

# F1 score
print(np.mean([f1_score(pic_y_test.reshape([-1, np.prod(y_tr_pred.shape[1:])])[i], (y_test_pred.reshape([-1, np.prod(y_tr_pred.shape[1:])])>=threshold)[i]) for i in range(pic_x_test.shape[0])]))
# 0.866449526991



## choose the best threshold based on the validation set on f1 score
# th_list = np.arange(0, 1, 0.05), 2017/5/7
th_list = np.arange(0, 1, 0.05) # 2018/4/21
f1_list = []
# th_best = 0.55 (original)
# th_best = 0.75 (2017/05/06)
th_best = 0
f1_score_best = 0

for th in th_list:
    f1_temp_list = []
    for i in range(len(pic_y_test)):
        f1_temp_list.append(f1_score(pic_y_test[i].flatten(),
                                     y_test_pred[i].flatten()>th))
    mean_score = np.mean(f1_temp_list)
    f1_list.append(mean_score)
    if mean_score > f1_score_best:
        f1_score_best = mean_score
        th_best = th

print('Best threshold:', th_best)
print('Best f1 socre:', f1_score_best)

""" old model
### Save the model
## reference: http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to YAML
model_yaml = autoencoder.to_yaml()
with open("D:/Project/Side_project_Electron_microscopy_segmentation/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
autoencoder.save_weights("D:/Project/Side_project_Electron_microscopy_segmentation/model.h5")
print("Saved model to disk")

### load json and create model
from keras.models import model_from_yaml
yaml_file = open('D:/Project/Side_project_Electron_microscopy_segmentation/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("D:/Project/Side_project_Electron_microscopy_segmentation/model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder = loaded_model
"""

### Save the model (2017/05/06)
## reference: http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to YAML
model_yaml = autoencoder.to_yaml()
with open("D:/Project/Side_project_Electron_microscopy_segmentation/model_4layers.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
autoencoder.save_weights("D:/Project/Side_project_Electron_microscopy_segmentation/model_4layers.h5")
print("Saved model to disk")
### 2018/04/21
# serialize model to YAML
model_yaml = autoencoder.to_yaml()
with open("D:/Project/Side_project_Electron_microscopy_segmentation/model_4layers_adam.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
autoencoder.save_weights("D:/Project/Side_project_Electron_microscopy_segmentation/model_4layers_adam.h5")
print("Saved model to disk")

### load json and create model
from keras.models import model_from_yaml
yaml_file = open('D:/Project/Side_project_Electron_microscopy_segmentation/model_4layers.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("D:/Project/Side_project_Electron_microscopy_segmentation/model_4layers.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder = loaded_model

### Show the model
from keras_sequential_ascii import sequential_model_to_ascii_printout
sequential_model_to_ascii_printout(autoencoder)

### Evaluate on the test dataset
### Load the data
test_img = imread('D:/Dataset/Side_project_Electron_microscopy_3d_segmentation/testing.tif')
test_img_truth = imread('D:/Dataset/Side_project_Electron_microscopy_3d_segmentation/testing_groundtruth.tif')

test_img = test_img/255
test_img_truth = test_img_truth/255

test_img = test_img.reshape([len(test_img), 768, 1024, 1])
test_img_truth = test_img_truth.reshape([len(test_img_truth), 768, 1024, 1])

y_pred = autoencoder.predict(test_img, batch_size=3)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# use the best threshold

# output from model
a_list = []
p_list = []
r_list = []
f_list = []
for i in range(len(y_pred)):
    a_list.append(accuracy_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))    
    p_list.append(precision_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))
    r_list.append(recall_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))
    f_list.append(f1_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))

print('Results on test dataset:')
print('Accuracy:', np.mean(a_list))
print('Precision:', np.mean(p_list))
print('Recall:', np.mean(r_list))
print('F1 score:', np.mean(f_list))
print('ROC AUC', np.mean([roc_auc_score(test_img_truth[i].flatten(), y_pred[i].flatten()) for i in range(y_pred.shape[0])]))

# baseline
a_list = []
p_list = []
r_list = []
f_list = []
for i in range(len(y_pred)):
    a_list.append(accuracy_score(test_img_truth[i].flatten(),
                                  test_img[i].flatten()>=threshold))    
    p_list.append(precision_score(test_img_truth[i].flatten(),
                                  test_img[i].flatten()>=threshold))
    r_list.append(recall_score(test_img_truth[i].flatten(),
                                  test_img[i].flatten()>=threshold))
    f_list.append(f1_score(test_img_truth[i].flatten(),
                                  test_img[i].flatten()>=threshold))

print('Results on test dataset:')
print('Accuracy:', np.mean(a_list))
print('Precision:', np.mean(p_list))
print('Recall:', np.mean(r_list))
print('F1 score:', np.mean(f_list))
print('ROC AUC', np.mean([roc_auc_score(test_img_truth[i].flatten(), test_img[i].flatten()) for i in range(y_pred.shape[0])]))



# use the 0.5 as threshold
threshold = 0.5
a_list = []
p_list = []
r_list = []
f_list = []
for i in range(len(y_pred)):
    a_list.append(accuracy_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))    
    p_list.append(precision_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))
    r_list.append(recall_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))
    f_list.append(f1_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>=threshold))

print('Results on test dataset:')
print('Accuracy:', np.mean(a_list))
print('Precision:', np.mean(p_list))
print('Recall:', np.mean(r_list))
print('F1 score:', np.mean(f_list))
print('ROC AUC', np.mean([roc_auc_score(test_img_truth[i].flatten(), y_pred[i].flatten()) for i in range(y_pred.shape[0])]))


# output a sample test data


th_best = 0.35

pic = 1
fig = plt.figure(figsize=(16, 14))
ax_1 = fig.add_subplot(2, 2, 1)
plt.imshow(test_img[pic].reshape([768, 1024]), cmap='gray')
ax_1.set_title('Raw Picture', fontsize=15)
plt.grid(False)

ax_2 = fig.add_subplot(2, 2, 2)
plt.imshow(y_pred[pic].reshape([768, 1024]), cmap='gray')
ax_2.set_title('Prediction on Test Set Using Tuned CNN Model', fontsize=15)
plt.grid(False)

ax_3 = fig.add_subplot(2, 2, 3)
plt.imshow((test_img[pic].reshape([768, 1024]))*(y_pred[pic].reshape([768, 1024])>th_best),
            cmap='gray')
ax_3.set_title('Map the Raw Data to Predicted Locations Using Trained Threshold', fontsize=15)
plt.grid(False)

ax_4 = fig.add_subplot(2, 2, 4)
plt.imshow(test_img_truth[pic].reshape([768, 1024]), cmap='gray')
ax_4.set_title('Ground Truth', fontsize=15)
plt.grid(False)