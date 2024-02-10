# MediaPipe-FSLR
**Filipino Sign Language Recognition using Support Vector Machines**

_A final project in AMAT 191: Introduction to Machine Learning_

## Rationale
This project aims to perform classification or recognition on 24 letters of the alphabet signed using Filipino Sign Language (FSL). Only 24 letters are considered which are in the form of static signs in FSL, omitting the letters J and Z.. The features from the image-formatted dataset will be extracted using MediaPipe Holistic, a pipeline developed by Google. Particularly, 21 hand landmarks are extracted from the images wherein each landmark is represented by its 3D coordinates (x,y,z). These landmarks are flattened to obtain the features of a single datapoint having 63 dimensions. These features will be relevant since it captures the hand shape, hand location, and palm orientation, essential in signed alphabet recognition. 

Once the features are gathered, the dataset is divided using a 80:20 train-test ratio. The training and testing is implemented using four (4) SVM models having different kernels (linear, RBF, poly, sigmoid). The performance of the models are compared by quantifying five (5) evaluation metrics: Precision, Recall, F1-score, Specificity, and Accuracy. 


## Framework
![image](https://github.com/ji-chani/MediaPipe-FSLR/assets/120572492/9a507c34-bb3d-42dc-b980-141c89f42a88)
