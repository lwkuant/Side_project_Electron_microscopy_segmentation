# Side Project: Electron Microscopy Segmentation Using Autoencoder and Convolutional Neural Network

## Overview
The goal of this peoject is to segment the targeted objects from the pictures under the electron microscopy.

The model used in this task is the autoencoder with convolutional neural network structure.

The related information regarding this dataset can be accessed below:
* [kaggle (the data source)](https://www.kaggle.com/kmader/electron-microscopy-3d-segmentation)

## Result
### 1. Structure of the model
<img src="https://github.com/lwkuant/Side_project_Electron_microscopy_segmentation/blob/master/model_overview.jpg">

### 2. The predicted outcome on one of the example from test dataset:
<img src="https://github.com/lwkuant/Side_project_Electron_microscopy_segmentation/blob/master/Compare_test_result_model_4layers.png">

### 3. Performance
| Metrics   	| Baseline 	| CNN+Autoencoder 	|
|-----------	|----------	|-----------------	|
| Accuracy  	| 0.085    	| 0.98            	|
| Precision 	| 0.05     	| 0.88            	|
| Recall    	| 0.92     	| 0.71            	|
| F1 score  	| 0.10     	| 0.79            	|
| AUC       	| 0.20     	| 0.99            	|

## Conclusions and takeaways
1. The results look not bad; however, the recall leaves some space for improvement since I think this measure is more important in this task
2. Before building, use some computer vision-specific preprocessing techniques to process the data to make it easier to find the patterns
3. treat this task as a object detection problem
4. Tune more hyperparameters

