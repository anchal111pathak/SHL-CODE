# Speech Grammar Scoring Using Machine Learning

## 📌 Project Overview

This project develops a machine learning model to predict grammar proficiency scores from spoken audio recordings. It was developed as part of the **SHL Hiring Assessment 2026** Kaggle competition.

The system extracts acoustic and speech-related features from audio recordings and uses an ExtraTrees Regressor to predict a continuous grammar score on a scale of **0 to 5**.

## 🎯 Objectives

* Process spoken audio recordings for machine learning.
* Extract meaningful acoustic features from speech.
* Train a regression model to predict grammar scores.
* Evaluate model generalization using five-fold cross-validation.
* Generate predictions for unseen evaluation recordings.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Audio Processing:** Librosa
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model:** ExtraTrees Regressor
* **Environment:** Kaggle Notebooks

## 🧠 Methodology

### 1. Data Preparation

The dataset contains labelled training audio recordings and unlabelled evaluation recordings.

* Training samples: 769
* Evaluation samples: 216
* Audio format: WAV
* Audio sampling rate: 16 kHz
* Target variable: Grammar score ranging from 0 to 5

### 2. Audio Feature Extraction

The following features are extracted from audio recordings:

* Mel-Frequency Cepstral Coefficients (MFCCs)
* MFCC mean and standard deviation
* Spectral centroid
* Spectral bandwidth
* Spectral rolloff
* Zero-crossing rate
* Root Mean Square (RMS) energy statistics
* Audio duration
* Waveform amplitude statistics

These features represent different acoustic properties of the speech signal.

### 3. Data Preprocessing

* Extract acoustic features from the recordings.
* Handle missing feature values using median imputation.
* Prepare feature matrices and corresponding target scores for training.
* Use consistent preprocessing during model training and prediction.

### 4. Model Training

An ExtraTrees Regressor is used for the regression task.

**Model configuration:**

* Number of estimators: 400
* Minimum samples per leaf: 2
* Maximum features: 0.8
* Random state: 42

The model is trained using the extracted audio features and their corresponding grammar scores.

### 5. Model Evaluation

Five-fold cross-validation is used to estimate performance on unseen training examples.

## 📊 Results

| Evaluation Metric   |          Result |
| ------------------- | --------------: |
| Training RMSE       |          0.1002 |
| Five-fold CV RMSE   | 0.7572 ± 0.0533 |
| Five-fold CV MAE    | 0.6044 ± 0.0421 |
| Kaggle Public Score |          0.7744 |

**Note:** Training RMSE measures error on the training data, whereas cross-validation metrics provide a more useful estimate of generalization. The substantial difference between training and cross-validation RMSE indicates a potential overfitting issue.

The Kaggle public score is reported as displayed on the competition page; its interpretation depends on the competition's scoring metric.

## 🔄 Project Workflow

1. Load training and evaluation metadata.
2. Read audio recordings.
3. Extract acoustic features using Librosa.
4. Impute missing feature values.
5. Train the ExtraTrees regression model.
6. Evaluate using five-fold cross-validation.
7. Retrain the selected model on the complete training dataset.
8. Generate predictions for evaluation recordings.
9. Save predictions in CSV format for Kaggle submission.

## 📁 Project Structure

```text
SHL-Grammar-Scoring-ML/
├── README.md
├── notebook3d13c8ce90.ipynb
└── submission_candidate.csv
```

*The filenames above are examples. Rename your notebook and include the CSV only if you intend to publish it and sharing is permitted by the competition rules.*

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/SHL-Grammar-Scoring-ML.git
cd SHL-Grammar-Scoring-ML
```

### 2. Install dependencies

```bash
pip install numpy pandas librosa scikit-learn
```

### 3. Open the notebook

Run the notebook in Jupyter or Kaggle after configuring the dataset paths to match your environment.

### 4. Generate predictions

Execute the feature extraction, model training, and prediction cells. The notebook saves the generated predictions as a CSV file.

## 🔍 Limitations and Future Improvements

* Explore additional speech and linguistic features.
* Compare ExtraTrees with gradient boosting and other regression models.
* Improve generalization through systematic hyperparameter tuning.
* Investigate speech recognition and text-based linguistic features.
* Analyze prediction errors across different grammar score ranges.
* Evaluate additional metrics and improve reproducibility.

## 💡 Key Learnings

* Audio feature engineering for machine learning.
* Regression modelling on speech-related data.
* Cross-validation for model evaluation.
* Audio preprocessing using Librosa.
* End-to-end machine learning workflow development.
* Generating and validating Kaggle competition submissions.

## 👩‍💻 Author

**Anchal Pathak**

AI Engineer | Machine Learning | Generative AI | Python



---

*This project was developed for the SHL Hiring Assessment 2026 competition. The reported metrics reflect the experiments and submission results obtained during development.*
