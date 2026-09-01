import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train_evaluate_knn(data_path: str, output_dir: str):
    print("Entrenando modelo KNN...")
    df = pd.read_csv(data_path)
    
    # Mapeo simplificado para modelado
    mappings = {
        'level': {'High School': 0, 'Undergraduate': 1},
        'gender': {'Male': 0, 'Female': 1},
        'zone.type': {'Rural':0, 'Urban':1, 'Semiurban':2},
        'school': {'EN': 0, 'ECMS': 1, 'EIC': 2, 'ECSG': 3, 'EHE': 4, 'EAAD': 5, 'High School': 6},
        'program': {'IC':0,'ICI':1,'ICT':2,'IDA':3,'IDS':4,'IFI':5,'IIA':6, 'IID':7,'IIN':8,'IIS':9,'IIT':10,'IMA':11,'IMD':12,'IME':13,'IMI':14,'IMT':15,'ING':16,'INQ':17,'INT':18,'IQA':19,'IQP':20,'ISC':21,'ISD':22,'ITC':23,'ITE':24,'ITI':25,'ITS':26,'IA':27,'IBT':28,'BIO':29,'IBQ':30,'TIE':31,'IBN':32,'LIN':33,'ARQ':34,'LDI':35,'LAE':36,'LAF':37,'LDE':38,'LEF':39,'LAD':40,'LMC':41},
        'region': {'RM':0,'RO':1,'RCM':2,'RCS':3,'DR':4},
        'foreign': {'Local':0, 'Yes':1},
        'scholarship.type': {'Academic talent':0,'Army/Navy scholarship':1, 'Child of Professor/Employee/Director':2,'Contigency scholarship':3, 'Cultural talent':4,'Enterpreneurial talent':5,'Leaders of Tomorrow Scholarship':6, 'Leadership talent':7,'No scholarship':8, 'Sports talent':9,'Traditional':10},
        'socioeconomic.level': {'Level 1': 0,'Level 2': 1,'Level 3': 2,'Level 4': 3,'Level 5': 4,'Level 6': 5,'Level 7': 6}
    }
    
    for col, mapping in mappings.items():
        df[col] = df[col].replace(mapping)

    X = df.drop(columns=["retention"])
    y = df["retention"]

    scaler = StandardScaler()
    clf = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='minkowski', p=2)

    accuracies, precisions, recalls, f1_scores, tprs, aucs = [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    confusion_matrices = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    plt.figure(figsize=(10, 10))

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold+1} (AUC = {roc_auc:.2f})')
        
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    print(f"Resultados Finales KNN:\nAccuracy: {np.mean(accuracies):.4f}\nPrecision: {np.mean(precisions):.4f}")

    # Guardar Matriz de Confusión
    os.makedirs(output_dir, exist_ok=True)
    cm_mean = np.mean(confusion_matrices, axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_mean, annot=True, fmt='.1f', cmap='Blues', xticklabels=['No', 'Si'], yticklabels=['No', 'Si'])
    plt.title('Matriz de Confusión Promedio - KNN')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_knn.png'))
    plt.close()

    # Guardar Curva ROC
    plt.figure(figsize=(10, 8))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'ROC promedio (AUC = {auc(mean_fpr, mean_tpr):.2f})', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
    plt.title('Curva ROC para KNN con Validación Cruzada')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve_knn.png'))
    plt.close()

if __name__ == "__main__":
    train_evaluate_knn(data_path="../data/dataset_preprocesado.csv", output_dir="../outputs")