# LandslideNet-EfficientNet-SVM

## Landslide Detection Using EfficientNetV2 and SVM

An open-source hybrid deep learning and machine learning project for landslide detection using remote sensing imagery. This work combines EfficientNetV2 as a deep feature extractor with a Support Vector Machine (SVM) classifier and addresses class imbalance using SMOTE. The approach is evaluated on the CAS Landslide Dataset.

---

## Project Overview

Landslides are a major natural hazard, causing severe damage to life and infrastructure. Accurate and automated detection from satellite and aerial imagery is essential for disaster prevention and mitigation. This project proposes a CNN–SVM hybrid framework that leverages deep visual features and robust classical machine learning techniques for reliable landslide detection.

---

## Key Features

- **EfficientNetV2-based** deep feature extraction
- **SVM classifier** with RBF kernel
- **Class imbalance handling** using SMOTE
- Evaluation using accuracy, F1-score, confusion matrix, and ROC curve
- **Deployment-ready** TensorFlow (`.h5`) and TensorFlow Lite (`.tflite`) models

---

## Methodology

1. Remote sensing image patches are preprocessed and normalized.
2. Deep features are extracted using a pretrained EfficientNetV2 backbone (frozen during training).
3. SMOTE is applied to balance the training dataset.
4. An SVM classifier performs binary classification (landslide vs non-landslide).
5. Model performance is evaluated on unseen test data.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | ~72% |
| F1-score | ~0.70 |

The results demonstrate stable and balanced performance on a challenging multi-region dataset with image-level labels derived from pixel-wise landslide masks.

---

## Repository Structure

```text
LandslideNet-EfficientNet-SVM/
├── Notebook/            # Jupyter notebooks for experiments
├── Training_Graphs/     # Accuracy, loss, ROC, and confusion matrix plots
├── Presentations/       # Project slides and figures
├── models/              # Saved models (.joblib, .h5, .tflite)
├── results/             # Metrics and evaluation outputs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset and Full Project Files

⚠️ **The full dataset and large intermediate files are not included in this repository due to size and licensing restrictions.**

### Access complete files here:

🔗 **Google Drive (Full Project Files):**  
[https://drive.google.com/drive/folders/1iTRbJ7NELWxOXlXSkjDGrJ2hafGkzxvt?usp=sharing](https://drive.google.com/drive/folders/1iTRbJ7NELWxOXlXSkjDGrJ2hafGkzxvt?usp=sharing)

This link contains:
- Processed datasets
- Full training outputs
- Additional notebooks and resources

Follow the notebooks in this repository to reproduce the results.

---

## Deployment

For deployment and demonstration purposes:

- The **EfficientNetV2 feature extractor** is provided in:
  - TensorFlow (`.h5`)
  - TensorFlow Lite (`.tflite`)
- The **SVM classifier** is saved separately as a `.joblib` file.

---

## Installation

```bash
pip install -r requirements.txt
```

**Recommended Python version:** 3.9 or higher

---

## License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this work with proper attribution.

---

## Contributions

Contributions, bug reports, and feature requests are welcome. Please fork the repository and submit a pull request for review.

---

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

---

⭐ **If you find this project useful, consider giving it a star on GitHub.**