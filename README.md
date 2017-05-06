# Side Project: Electron Microscopy Segmentation Using Autoencoder and Convolutional Neural Network

## Overview
The goal of this peoject is to segment the targeted objects from the pictures under the electron microscopy.

The model used in this task is the autoencoder with convolutional neural network structure.

The related information regarding this dataset can be accessed below:
* [kaggle (the data source)](https://www.kaggle.com/kmader/electron-microscopy-3d-segmentation)

## Result
### Structure of the model
<img src="https://github.com/lwkuant/Side_project_Electron_microscopy_segmentation/blob/master/model_overview.jpg">

### The predicted outcome on one of the example from test dataset:
<img src="https://github.com/lwkuant/Side_project_Electron_microscopy_segmentation/blob/master/Compare_test_result_model_4layers.png">

### The result looks not bad, it achieves:
* Accuracy: 0.98
* Precision: 0.86
* Recall: 0.77
* F1 score: 0.81

## Conclusions and takeaways
1. The results look not bad; however, the recall leaves some space for improvement since I think this measure is more important in this task
2. Before building, use some computer vision-related preprocessing techniques to process the data to make it easier to find the patterns
3. Try different structures of the model
4. Tune more hyperparameters 

