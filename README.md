# Side Project: Electron Microscopy Segmentation Using Autoencoder and Convolutional Neural Network

## Overview
The goal of this peoject is to segment the targeted objects from the pictures under the electron microscopy.

The model used in this task is the autoencoder with convolutional neural network structure.

The related information regarding this dataset can be accessed below:
* [kaggle (the data source)](https://www.kaggle.com/kmader/electron-microscopy-3d-segmentation)

## Result
### The predicted outcome on one of the example from test dataset:
<img src="https://github.com/lwkuant/Side_project_Electron_microscopy_segmentation/blob/master/Compare_test_result.png">

### The result looks not bad, it achieves:
* Accuracy: 0.98
* Precision: 0.82
* Recall: 0.75
* F1 score: 0.78

## Further
* Use some effective preprocessing techniques
* Tune the structure to better remove the False Positive predictions (the actual outcome is no while the model's prediction is true)
