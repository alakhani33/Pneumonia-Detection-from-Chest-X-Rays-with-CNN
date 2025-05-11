
# 🩻 Pneumonia Detection from Chest X-Rays using CNN (Transfer Learning)

This project builds a deep learning pipeline to detect **pneumonia** from chest X-ray scans using a **Convolutional Neural Network (CNN)** based on **ResNet50**. It leverages transfer learning and data augmentation techniques to achieve high performance on a well-known open dataset.

---

## 📁 Dataset

We use the **Chest X-Ray Pneumonia** dataset from Kaggle:  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset structure after extraction:

```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

---

## 🚀 Features

- 📊 Binary classification: **Pneumonia** vs **Normal**
- 🧠 Transfer Learning using **ResNet50**
- 🎛 Real-time **data augmentation** with `ImageDataGenerator`
- 📈 Training history plots
- 🔍 Model evaluation on test set
- 🔧 Fine-tuning the CNN for improved accuracy

---


## 🧪 Model Architecture

- **Base:** ResNet50 (ImageNet pre-trained, frozen initially)
- **Custom Head:**
  - GlobalAveragePooling
  - Dense(128, relu)
  - Dense(1, sigmoid) for binary classification

---

## 📉 Performance Evaluation

The notebook reports:

- Training & validation accuracy over epochs
- Test accuracy before and after fine-tuning

---
