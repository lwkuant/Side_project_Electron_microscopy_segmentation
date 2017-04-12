# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 01:32:38 2017

@author: user
"""

### Load required packages
import pandas as pd 
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from skimage.io import imread

### Load the data
train_img = imread('D:/Dataset/Side_project_Electron_microscopy_3d_segmentation/training.tif')
train_img_truth = imread('D:/Dataset/Side_project_Electron_microscopy_3d_segmentation/training_groundtruth.tif')

### show an example
pic = 100 
fig = plt.figure(figsize=(15, 4))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(train_img[pic], cmap='gray')

ax_2 = fig.add_subplot(1, 3, 2)
plt.hist(train_img[pic].flatten())

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(train_img_truth[pic], cmap='gray')

### show the segmentation 
fig = plt.figure(figsize=(12, 10))
ax_1 = fig.add_subplot(2, 2, 1)
plt.imshow(train_img[pic], cmap='gray')
ax_2 = fig.add_subplot(2, 2, 2)
plt.hist(train_img[pic].flatten())

ax_3 = fig.add_subplot(2, 2, 3)
plt.imshow(train_img_truth[pic]*train_img[pic], cmap='gray')
ax_4 = fig.add_subplot(2, 2, 4)
#plt.hist((train_img_truth[pic]*train_img[pic]).flatten())
plt.hist((train_img[pic][train_img_truth[pic]>0]))


### Using the threshold from histogram to segment
minimum = (train_img[pic][train_img_truth[pic]>0]).min()
maximum = (train_img[pic][train_img_truth[pic]>0]).max()
fig = plt.figure(figsize=(15, 4))
ax_1 = fig.add_subplot(1, 3, 1)
plt.imshow(train_img[pic], cmap='gray')

ax_2 = fig.add_subplot(1, 3, 2)
plt.imshow((train_img[pic]>minimum)&(train_img[pic]<maximum), cmap='gray')

ax_3 = fig.add_subplot(1, 3, 3)
plt.imshow(train_img_truth[pic]>0, cmap='gray')
# loks not good enough...

### Evaluate the sgementation (on the benchmark segmentation)
y = ((train_img_truth[pic]>0)*1).flatten()
y_pred = (1 - train_img[pic]/255).flatten()

from sklearn.metrics import roc_auc_score, roc_curve, auc

fpr, tpr, th = roc_curve(y, y_pred)
score = roc_auc_score(y, y_pred)

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

ind = np.random.choice(range(train_img.shape[0]), int(train_img.shape[0]*0.8),
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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

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


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(pic_x_tr, pic_y_tr,
                epochs=100,
                batch_size=3,
                shuffle=True,
                validation_data=(pic_x_test, pic_y_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

## choose the best threshold based on the validation set on f1 score
th_list = np.arange(0, 1, 0.05)
f1_list = []
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

### Save the model
## reference: http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to YAML
model_yaml = autoencoder.to_yaml()
with open("D:/Project/Side_project_Electron_microspy_segmentation/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
autoencoder.save_weights("D:/Project/Side_project_Electron_microspy_segmentation/model.h5")
print("Saved model to disk")


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

a_list = []
p_list = []
r_list = []
f_list = []
for i in range(len(y_pred)):
    a_list.append(accuracy_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>th_best))    
    p_list.append(precision_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>th_best))
    r_list.append(recall_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>th_best))
    f_list.append(recall_score(test_img_truth[i].flatten(),
                                  y_pred[i].flatten()>th_best))

print('Results on test dataset:')
print('Accuracy:', np.mean(a_list))
print('Precision:', np.mean(p_list))
print('Recall:', np.mean(r_list))
print('F1 score:', np.mean(f_list))

pic = 100
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