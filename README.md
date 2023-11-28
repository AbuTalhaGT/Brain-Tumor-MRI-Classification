# Brain-Tumor-MRI-Classification
## Project Title : Project Title: Intelligent Brain Tumor MRI Classification using CNN 

## Objective : Medical imaging, especially MRI for the brain, is crucial in diagnosing and treating diseases. This project utilizes Convolutional Neural Networks and artificial intelligence for classifying brain tumors in MRI scans.

## What is a brain tumor?

A brain tumor is a collection, or mass, of abnormal cells in your brain. Your skull, which encloses your brain, is very rigid. Any growth inside such a restricted space can cause problems. Brain tumors can be cancerous (malignant) or noncancerous (benign). When benign or malignant tumors grow, they can cause the pressure inside your skull to increase. This can cause brain damage, and it can be life-threatening.

## The importance of the subject

Early detection and classification of brain tumors is an important research domain in the field of medical imaging and accordingly helps in selecting the most convenient treatment method to save patients life therefore.

## Basic Requirements

| Library                | Version                                    |
|------------------------|--------------------------------------------|
| Python                 | 3.7.10 | packaged by conda-forge | (default, Feb 19 2021, 16:07:37) [GCC 9.3.0] |
| TensorFlow            | 2.4.1                                      |
| Keras                  | 2.4.3                                      |
| Keras Preprocessing   | 1.1.2                                      |
| Matplotlib            | 3.4.1                                      |
| OpenCV                | 4.5.1                                      |
| scikit-learn          | 0.24.1                                     |


## Dataset

We have referred the following Research Paper and the corresponding dataset used in the paper
[Paper: Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks](https://www.mdpi.com/2076-3417/10/6/1999)

The datasets utilized in this paper are Brain Tumor MRI dataset Msoud and we also used the same. [here](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset).

### Dataset Details

This dataset contains **7022** images of human brain MRI images which are classified into 4 classes:

- glioma
- meningioma
- no tumor
- pituitary

But we have used subset of the above dataset, which consists of 2870 images in the training data and 394 images in testing data. We did this because of limited computational resources (GPUs and RAM), and run our model in Kaggle Notebook.

### Data Pre-processing

We performed the following pre-processing steps :

- Normalization 
- Data Augmentation
- Convert Vector Classes to Binary Class Matrices 

## Custom CNN Model 

We have created a baseline CNN model which consists of 3 Convolution and Pooling layers, in addition to this one more custom CNN model has created consisting of 6 Convolution and Pooling layers.

## Pre-trained Model

A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. Accordingly, due to the computational cost of training such models, it is common practice to import and use models from published literature . For this project, I decided to use **ResNet50** model to perform image classification for brain tumor MRI images.[Resnet50 Article](https://arxiv.org/abs/1512.03385)

## Training 

We have trained baseline CNN model for 30 epochs, custom CNN (model -2) for 10 epochs and Resnet50 for 50 epochs for this project. 

## Evaluation

After completion of training , we have predicted our results on testing dataset.

## Metrics Used

- Accuracy Definition - Accuracy is a common evaluation metric that measures the overall correctness of the model across all classes

- Accuracy = Total Number of Predictions / Number of Correct Predictions (For Baseline Model and ResNet50)

- Categorical Accuracy for a Class =  Total Number of Actual Instances of that Class / Number of Correct Predictions for that Class (For Custom CNN_Model-2)
​
- Categorical Accuracy Definition - Categorical accuracy, also known as categorical accuracy or categorical cross-entropy accuracy, is specifically used in multi-class classification problems.
​

## Results
- Baseline CNN (Model-1) : Training Accuracy - 77.2%, Testing Accuracy - 75.26%
- Custom CNN (Model-2) : Training Accuracy - 98.39%, Testing Accuracy - 84.67%
- ResNet50 (Pre-trained model) : Training Accuracy - 64.20%, Testing Accuracy - 75.26% (Which is not a good indication because testing accuracy is more than training dataset)

We are still in the process to achieve the 
## Contributing

Thank you for considering contributing to the Brain Tumor MRI Classification project! Your involvement is essential for the success of this project and the improvement of its capabilities.

### How to Contribute

1. Fork the Repository: Start by forking the project on GitHub.
2. Clone the Repository: Clone the forked repository to your local machine using the following command:
   ```bash
   git clone https://github.com/AbuTalhaGT/Brain-Tumor-MRI-Classification.git

## Note
You can see more details about training steps and testing results inside [Group_8_Brain_Tumor_MRI.ipynb](https://github.com/AbuTalhaGT/Brain-Tumor-MRI-Classification/blob/main/Group_8_Brain_Tumor_MRI.ipynb)

