# Fabric Defect Detection System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ashen97/fabric-defect-live-stream)

Check out the live demo here: [**Live App**](https://ashen97-fabric-defect-live-stream.hf.space)

This repository contains the full project pipelie for fabric defect detection system, developed as part of an MSc in Data Science research project. The system compares traditional Machine Learning (SVM) with Deep Learning (YOLOv8, YOLOv11) approaches to identify defects in textile fabrics in real-time.

## 📌 Project Overview

The primary goal of this project is to develop a computer vision solution capable of detecting fabric defects (e.g., holes, stains, yarn misalignments) on industrial production lines. The study addresses the challenge of **Class Imbalance** (a 3:1 ratio of normal to defective samples) using a physical Random Oversampling strategy.

**Key Features:**
* **Baseline Model:** Histogram of Oriented Gradients (HOG) + Linear SVM.
* **Deep Learning Models:** YOLOv8n, YOLOv8s, and YOLOv11n.
* **Data Optimization:** Random oversampling to balance the dataset.
* **Performance:** The optimal model (**YOLOv8n**) achieves **68.3% Recall** and **85.7% Precision**.

## 📂 Repository Structure

The code is organized into sequential notebooks for reproducibility:

| File Name | Description |
| :--- | :--- |
| `data_annotation.ipynb` | Scripts for converting dataset annotations. |
| `data_splitting.ipynb` | Splits the raw dataset into Training, Validation, and Test sets. |
| `eda_process.ipynb` | Exploratory Data Analysis (EDA) to visualize class distribution. |
| `model_train_svm.ipynb` | Training and evaluation of the baseline HOG + SVM model. |
| `model_train_yolo_oversampling.ipynb`| Implements the Random Oversampling strategy and trains the YOLO models. |
| `model_train.ipynb` | Initial training scripts for baseline YOLO models without balancing. |
| `prediction_demo.ipynb` | Inference script to real-time defect detection. |

## 📊 Dataset Reference

This research used the **ZJU-Leaper Dataset**, a benchmark dataset for fabric defect detection collected by researchers at **Zhejiang University**. We gratefully acknowledge their contribution to the open-source community.

**Dataset API:**
https://github.com/nico-zck/ZJU-Leaper-Dataset/tree/master/dataset_api

**Dataset Citation:**
If you use this code or dataset, please cite the original paper:

```bibtex
@ARTICLE{9346038,
  author={Zhang, Chenkai and Feng, Shaozhe and Wang, Xulongqi and Wang, Yueming},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={ZJU-Leaper: A Benchmark Dataset for Fabric Defect Detection and a Comparative Study}, 
  year={2020},
  volume={1},
  number={3},
  pages={219-232},
  keywords={Fabrics;Inspection;Automatic optical inspection;Benchmark testing;Textile industry;Fabric defect detection;automatic inspection;dataset;benchmark},
  doi={10.1109/TAI.2021.3057027}}
````

## 🚀 Installation & Usage


1. Clone the repository:
```
git clone https://github.com//ashen-pabasara/fabric-defect-detection.git
```
2. Install Dependencies: This project requires Python 3.8+ and the Ultralytics library.
```
pip install ultralytics opencv-python-headless scikit-learn pandas matplotlib
```
3. Run the Training: Open `model_train_yolo_oversampling.ipynb` in Jupyter Notebook or Google Colab to reproduce the balanced training results.

4. Run Inference: Use ``prediction_demo.ipynb`` to test the model on your own images.

## 📈 Results Summary

| Model Configuration | Recall | Precision | Inference Speed |
| :--- | :--- | :--- | :--- |
| SVM (Baseline) | 45.0% | 36.0% | N/A |
| YOLOv8n (Imbalanced) | 63.4% | 81.5% | 1.5 ms |
| YOLOv8n (Balanced) | 68.3% | 85.7% | 1.5 ms |
| YOLOv8s (Balanced) | 69.0% | 84.6% | 3.3 ms |
| YOLOv11n (Balanced) | 68.4% | 81.5% | 1.4 ms |
