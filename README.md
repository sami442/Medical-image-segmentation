# 🧠 Brain MRI Tumor Segmentation using U-Net

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview
A deep learning project that performs pixel-level segmentation of 
brain tumors in MRI scans using the U-Net architecture. 
This tool assists in early and accurate tumor detection, 
supporting healthcare professionals in diagnosis.

## 🎯 Objectives
- Detect and segment tumor regions in brain MRI images
- Achieve high accuracy using U-Net architecture
- Visualize segmentation masks over original MRI scans

## 🗂️ Dataset
- **Source:** [LGG Brain MRI Segmentation - Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Contents:** 110 patients, 3929 MRI images with tumor masks

## 🧠 Model Architecture
- **Base Model:** U-Net
- **Encoder:** Contracting path with Conv2D + MaxPooling
- **Decoder:** Expansive path with UpSampling + Skip Connections
- **Loss Function:** Binary Crossentropy + Dice Loss
- **Optimizer:** Adam

## 📊 Results
| Metric | Score |
|--------|-------|
| Accuracy | 99.37% |
| IoU Score | - |
| Dice Coefficient | 0.3147 |

## 📈 Results Visualization

![Training History](https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/training_history_v2.png)

![Predictions](https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/predictions_v2.png)

## 🛠️ Technologies Used
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## 🚀 How to Run
1. Clone the repository
```bash
git clone https://github.com/sami442/medical-image-segmentation.git
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Open the notebook
```bash
jupyter notebook notebooks/unet_segmentation.ipynb
```

## 👩‍💻 Author
**Samina Mazhar**
 - GitHub: [@sami442](https://github.com/sami442)

## 📄 License
This project is licensed under the MIT License.
- **Format:** PNG images + corresponding binary masks

## 🏗️ Project Structure
