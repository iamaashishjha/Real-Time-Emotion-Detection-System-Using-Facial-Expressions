# Emotion detection using deep learning

## Introduction
****
This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies

* [Python 3](https://www.python.org/), [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/) and many more libraries
* To install the required packages, run `pip install -r requirements.txt`.

## Basic Usage

* First, clone the repository and enter the folder

```bash
git clone https://github.com/iamaashishjha/Real-Time-Emotion-Detection-System-Using-Facial-Expressions.git
cd Real-Time-Emotion-Detection-System-Using-Facial-Expressions
```

* Download the FER-2013 dataset (train and test directory) inside the `data` folder.

* If you want to train this model, use:  

```bash
python main.py --mode train // To Train the model

```

<!-- * If you want to view the predictions without training again, you can download the pre-trained model from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view?usp=sharing) and then run:   -->
* If you want to run this app, use:  

```bash
python main.py // To Run the App
```

* The folder structure is of the form: 
  * data (folder)
  * `main.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model/model.h5` (file)
  * `imgs/icon.ico` (image)

## Data Preparation (optional)

* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013) is available as a  dataset of images in the PNG format for training/testing.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.
