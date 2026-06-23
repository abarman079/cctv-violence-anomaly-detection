# CCTV Violence Anomaly Detection

## Project Overview

This project presents a CCTV-based violence anomaly detection system using computer vision and machine learning. The goal is to classify surveillance-style images as either normal/non-violent or anomalous/violent.

The project compares multiple machine learning and deep learning models, including classical feature-based classifiers, a small custom CNN, and a MobileNetV2 transfer learning model. The final results show that the MLP Classifier achieved the strongest overall F1-score among the tested models.

## Dataset

The project uses the **Violence vs. Non-Violence: 11K Images Dataset**, which contains violent and non-violent scene images.

For this experiment, the dataset was prepared as a binary classification problem:

| Label | Class                 |
| ----- | --------------------- |
| 0     | Normal / Non-Violence |
| 1     | Anomaly / Violence    |

The project used a balanced subset of 5,000 images:

| Class   | Image Count | Percentage |
| ------- | ----------: | ---------: |
| Normal  |       2,500 |        50% |
| Anomaly |       2,500 |        50% |

The data was split into:

| Split      | Images |
| ---------- | -----: |
| Training   |  3,500 |
| Validation |    750 |
| Test       |    750 |

## Project Workflow

The project follows this workflow:

1. Load and inspect the CCTV violence/non-violence image dataset.
2. Balance the normal and anomaly classes.
3. Resize images to 160 × 160 pixels.
4. Split the data into training, validation, and test sets.
5. Extract features for classical machine learning models.
6. Train multiple baseline classifiers.
7. Train deep learning models using CNN and MobileNetV2 transfer learning.
8. Evaluate all models using accuracy, precision, recall, F1-score, ROC-AUC, and average precision.
9. Generate visualizations, confusion matrices, ROC curves, precision-recall curves, and alert demo predictions.
10. Compare final model performance.

## Models Compared

The project compares the following models:

* MLP Classifier
* MobileNetV2 Transfer Learning
* KNN Classifier
* Random Forest
* Logistic Regression
* Linear SVM
* Small CNN
* Isolation Forest

## Final Model Performance

| Model                | Accuracy | Precision | Recall / TPR | F1-Score | ROC-AUC | Avg. Precision | Training Time |
| -------------------- | -------: | --------: | -----------: | -------: | ------: | -------------: | ------------: |
| MLP Classifier       |   94.93% |    94.23% |       95.73% |   94.97% |  99.10% |         99.14% |         3.26s |
| MobileNetV2 Transfer |   94.00% |    91.67% |       96.80% |   94.16% |  98.38% |         98.40% |        43.92s |
| KNN Classifier       |   94.00% |    91.88% |       96.53% |   94.15% |  98.29% |         97.55% |         0.32s |
| Random Forest        |   92.13% |    90.72% |       93.87% |   92.27% |  97.67% |         97.59% |         5.77s |
| Logistic Regression  |   91.60% |    90.84% |       92.53% |   91.68% |  98.12% |         98.19% |         0.90s |
| Linear SVM           |   91.20% |    90.77% |       91.73% |   91.25% |  97.57% |         97.65% |         1.73s |
| Small CNN            |   75.20% |    80.39% |       66.67% |   72.89% |  81.11% |         82.19% |        26.10s |
| Isolation Forest     |   56.93% |    64.29% |       31.20% |   42.01% |  62.55% |         62.46% |         0.47s |

## Key Findings

The MLP Classifier achieved the best F1-score and overall performance. MobileNetV2 Transfer Learning also performed strongly, especially for anomaly recall, but required more training time. KNN performed competitively with very low training time.

The Small CNN and Isolation Forest performed weaker than the supervised models. This suggests that feature quality and model selection were important for this binary violence anomaly detection task.

## Repository Structure

```text
docs/report/        Final project report in DOCX and PDF format
notebooks/          Main Jupyter Notebook
results/figures/    Generated plots, curves, confusion matrices, and demo images
results/tables/     Model metrics, classification reports, split summaries, and logs
models/             Saved trained Keras models
artifacts/features/ Saved NumPy feature and label arrays
data/               Dataset note and download instructions
```

## Important Files

* `notebooks/cctv-violence-anomaly-code.ipynb`
  Main notebook containing dataset loading, preprocessing, model training, evaluation, and visualization.

* `results/tables/final_model_comparison.csv`
  Final comparison of all tested models.

* `results/figures/figure_11_final_model_comparison_f1.png`
  F1-score comparison chart.

* `results/figures/figure_12_final_model_comparison_recall.png`
  Recall comparison chart.

* `results/figures/figure_13_alert_demo_predictions.png`
  Demonstration of model-based alert predictions.

* `models/mobilenetv2_transfer_best.keras`
  Saved MobileNetV2 transfer learning model.

* `models/small_cnn_best.keras`
  Saved custom CNN model.

## Technologies Used

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* MobileNetV2
* Computer Vision
* Transfer Learning

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/abarman079/cctv-violence-anomaly-detection.git
cd cctv-violence-anomaly-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it according to the instructions in `data/README.md`.

4. Open the notebook:

```bash
jupyter notebook notebooks/cctv-violence-anomaly-code.ipynb
```

5. Run the notebook cells in order.

## Notes

Large model files and NumPy arrays should be tracked with Git LFS if they become too large for normal GitHub storage.

## Ethics and Limitations

This project is for academic and research purposes only. It should not be used as a real-world surveillance, policing, or security decision system without additional testing, fairness evaluation, privacy review, and human oversight.

The model was trained and evaluated on a limited image dataset. Real CCTV violence detection usually requires video-level temporal modeling, scene variation testing, and robust evaluation on unseen camera environments.

## Author

Abarman079
