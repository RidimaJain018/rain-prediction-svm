# Rain Prediction with SVM

Binary classification to predict whether it will rain tomorrow using Support Vector Machine (RBF kernel) on the Australian Weather Dataset.

## Dataset
- Source: [Weather Dataset – Rattle Package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- 145,460 records | 23 features | Target: `RainTomorrow`

## What's in this project
- EDA and missing value handling
- Feature engineering: `PressureChange`, `HumidityDrop`, `TempRange`
- Location clustering to replace raw `Location` column
- Seasonal feature extraction from `Date`
- Encoding: binary mapping, OHE for wind directions and seasons
- Class imbalance handling with SMOTE
- StandardScaler (mandatory for SVM)
- SVC with RBF kernel, C=1

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 82.8% |
| Precision (Rain) | 0.60 |
| Recall (Rain) | 0.71 |
| F1 (Rain) | 0.65 |

## Why RBF?
Weather patterns are not linearly separable in the original feature space. RBF maps data to a higher-dimensional space where a clean margin can be found.

## Key Lesson
Scaling is non-negotiable with SVM — the model works on distances, so unscaled features distort the margin entirely.

## Requirements
```
scikit-learn
imbalanced-learn
pandas
numpy
seaborn
matplotlib
```
