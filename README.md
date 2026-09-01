# Student Dropout Prediction - Data Science Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)

## Project's Description
This repo contains the project I developed for the data science subject, which consists on the prediction on student dropout. The project includes from the cleaning and data wrangling (imputation and class balance) to the training and eval on prediction models (KNN & Multilayer Neural Networks), using cross validation to ensure the quality of the model.

> ** (NDA):**
> The code and the architecture is open, but the main purpose of the repo is to show my abilities in ML and DS, and this project is under a confidentiality agreement on the dataset, due to that, the dataset is not included. 

## Structure

\`\`\`
├── data/
├── src/
│   ├── 01_preprocessing.py       
│   ├── 02_model_knn.py            
│   └── 03_model_neural_network.py 
├── outputs/                       
├── requirements.txt               
└── README.md                      
\`\`\`

## Metodology

The project development was divided into three main phases:

1. **Advanced Preprocessing (`01_preprocessing.py`):**
- Handling of missing values ​​using nearest-neighbor-based imputation (`fancyimpute.KNN`). 
- Transformation of categorical variables (gender, socioeconomic status, scholarship type, etc.). 
- Addressing class imbalance using the **SMOTEENN** hybrid technique (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbours), enhancing the model's ability to detect the minority class (dropout).

2. **Baseline Model - K-Nearest Neighbors (`02_model_knn.py`):**
- Implementation of an optimized KNN classifier. 
- Evaluation via cross-validation (K-Fold, k=5). 
- Generation of confusion matrices and ROC curves.

3. **Deep Model - Artificial Neural Network (`03_model_neural_network.py`):**
- Multilayer Perceptron (MLPClassifier) ​​architecture with tuned hidden layers (200, 100, 50). 
- Training using the Adam optimizer, L2 regularization (Alpha=0.0001), and early stopping to prevent overfitting. 
- Rigorous evaluation using K-Fold (k=10) and statistical significance tests (t-test) to validate that performance exceeds chance levels.

## 📊 Results and metrics 
The scripts automatically generate performance plots in the `outputs/` folder. The metrics evaluated for both models include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Area Under the ROC Curve (AUC-ROC)**

## 👨‍💻 Autor
**Jesús Elian Peralta Chávez**
