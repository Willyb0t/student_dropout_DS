import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

def train_evaluate_nn(data_path: str, output_dir: str):
    print("Entrenando modelo de Red Neuronal (MLP)...")
    df = pd.read_csv(data_path)

    categorical_columns = ["level","gender","foreign", "zone.type", "school", "program", "region", "scholarship.type", "socioeconomic.level"]
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = onehot_encoder.fit_transform(df[categorical_columns])

    X_encoded = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(categorical_columns))
    X_encoded.index = df.index
    y = df['retention']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scaler = StandardScaler()

    nn_classifier = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        verbose=False 
    )

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    predictions, actual_values, predictions_proba = [], [], []
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    for fold, (train_index, test_index) in enumerate(kf.split(X_encoded)):
        X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        nn_classifier.fit(X_train_scaled, y_train)
        y_pred = nn_classifier.predict(X_test_scaled)
        y_pred_proba = nn_classifier.predict_proba(X_test_scaled)[:, 1]
        
        predictions.extend(y_pred)
        actual_values.extend(y_test)
        predictions_proba.extend(y_pred_proba)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        aucs.append(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    t_stat, p_value = stats.ttest_1samp(accuracies, 0.5)

    print("\nResultados de la Validación Cruzada (k=10):")
    print(f"Accuracy:  {np.mean(accuracies):.4f}")
    print(f"p-value (vs Random): {p_value:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    # Matriz de Confusión
    cm = confusion_matrix(actual_values, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Red Neuronal')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_neural_network.png'))
    plt.close()

    # Curva ROC
    plt.figure(figsize=(10, 8))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'ROC promedio (AUC = {auc(mean_fpr, mean_tpr):.2f})', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
    plt.title('Curva ROC con Validación Cruzada\nRed Neuronal')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve_neural_network.png'))
    plt.close()

if __name__ == "__main__":
    train_evaluate_nn(data_path="../data/dataset_preprocesado.csv", output_dir="../outputs")