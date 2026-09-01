import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
from imblearn.combine import SMOTEENN

# Definición de diccionarios de mapeo
DICT_LEVEL = {'High School': 0, 'Undergraduate': 1, 'Does not apply ': 2}
DICT_GENDER = {'Male': 0, 'Female': 1, 'Does not apply ': 2}
DICT_ZONE = {'Rural': 0, 'Urban': 1, 'Semiurban': 2, 'Does not apply ': 3}
DICT_SCHOOL = {'EN': 0, 'ECMS': 1, 'EIC': 2, 'ECSG': 3, 'EHE': 4, 'EAAD': 5, 'High School': 6, 'Does not apply ': 7}
DICT_PROGRAM = {'IC':0,'ICI':1,'ICT':2,'IDA':3,'IDS':4,'IFI':5,'IIA':6, 'IID':7,'IIN':8,'IIS':9,'IIT':10,'IMA':11,'IMD':12,'IME':13,'IMI':14,'IMT':15,'ING':16,'INQ':17,'INT':18,'IQA':19,'IQP':20,'ISC':21,'ISD':22,'ITC':23,'ITE':24,'ITI':25,'ITS':26,'IA':27,'IBT':28,'BIO':29,'IBQ':30,'TIE':31,'IBN':32,'LIN':33,'ARQ':34,'LDI':35,'LAE':36,'LAF':37,'LDE':38,'LEF':39,'LAD':40,'LMC':41, 'Does not apply ': 42}
DICT_REGION = {'RM':0,'RO':1,'RCM':2,'RCS':3,'DR':4, 'Does not apply ': 5}
DICT_FOREIGN = {'Local':0, 'Yes':1, 'Does not apply ': 2}
DICT_SCHOLARSHIP = {'Academic talent':0,'Army/Navy scholarship':1, 'Child of Professor/Employee/Director':2,'Contigency scholarship':3, 'Cultural talent':4,'Enterpreneurial talent':5,'Leaders of Tomorrow Scholarship':6, 'Leadership talent':7,'No scholarship':8, 'Sports talent':9,'Traditional':10, 'Does not apply ': 11}
DICT_SOCIOECONOMIC = {'Level 1': 0,'Level 2': 1,'Level 3': 2,'Level 4': 3,'Level 5': 4,'Level 6': 5,'Level 7': 6, 'Does not apply ': 7}

def preprocess_data(input_path: str, output_path: str):
    print("Iniciando preprocesamiento de datos...")
    df_tec = pd.read_csv(input_path)

    # Eliminar columnas no relevantes
    columns_to_drop = ['max.degree.parents', 'generation', 'educational.model', 'social.lag', 'first.generation',
                       'admission.rubric', 'total.scholarship.loan', 'id.school.origin', 'school.cost', 'tec.no.tec',
                       'father.education.complete', 'father.education.summary', 'mother.education.complete',
                       'mother.education.summary', 'parents.exatec', 'father.exatec', 'mother.exatec',
                       'scholarship.perc', 'loan.perc', 'average.first.period', 'physical.education', 'cultural.diffusion',
                       'student.society', 'total.life.activities', 'athletic.sports', 'art.culture',
                       'student.society.leadership', 'life.work.mentoring', 'wellness.activities', 'student.id']
    df_tec.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Seleccionar datos de la clase 'EIC'
    df_eic = df_tec[df_tec['school'] == 'EIC'].copy()
    print(f"Cantidad de datos de la clase 'EIC': {len(df_eic)}")

    # Limpieza y estandarización
    df_eic.replace(["No information", "Does not apply "], np.nan, inplace=True)
    if 'foreign' in df_eic.columns:
        df_eic['foreign'] = df_eic['foreign'].replace({'Yes: National': 'Yes', 'Yes: Foreigner': 'Yes'})

    # Mapeo a valores numéricos
    df_eic['gender'] = df_eic['gender'].replace(DICT_GENDER)
    df_eic['level'] = df_eic['level'].replace(DICT_LEVEL)
    df_eic['zone.type'] = df_eic['zone.type'].replace(DICT_ZONE)
    df_eic['school'] = df_eic['school'].replace(DICT_SCHOOL)
    df_eic['program'] = df_eic['program'].replace(DICT_PROGRAM)
    df_eic['region'] = df_eic['region'].replace(DICT_REGION)
    df_eic['foreign'] = df_eic['foreign'].replace(DICT_FOREIGN)
    df_eic['scholarship.type'] = df_eic['scholarship.type'].replace(DICT_SCHOLARSHIP)
    df_eic['socioeconomic.level'] = df_eic['socioeconomic.level'].replace(DICT_SOCIOECONOMIC)

    # Imputación KNN
    knn_imputer = KNN()
    df_eic_imputed = knn_imputer.fit_transform(df_eic)
    df_eic_imputed_dt = pd.DataFrame(df_eic_imputed, columns=df_eic.columns)
    
    print("Porcentaje de datos faltantes tras imputación:\n", df_eic_imputed_dt.isnull().mean() * 100)

    # Resampling con SMOTEENN
    X = df_eic_imputed_dt.drop(columns=['retention'])
    y = df_eic_imputed_dt['retention']
    
    smote_enn = SMOTEENN(random_state=42, sampling_strategy=0.3)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['retention'] = y_resampled

    # Redondear y revertir codificación para guardar el CSV de forma legible
    categorical_columns = ['gender', 'level', 'zone.type', 'school', 'program', 
                          'region', 'foreign', 'scholarship.type', 'socioeconomic.level']
    for col in categorical_columns:
        df_resampled[col] = df_resampled[col].round()

    reverse_dicts = {
        'gender': {v: k for k, v in DICT_GENDER.items()},
        'level': {v: k for k, v in DICT_LEVEL.items()},
        'zone.type': {v: k for k, v in DICT_ZONE.items()},
        'school': {v: k for k, v in DICT_SCHOOL.items()},
        'program': {v: k for k, v in DICT_PROGRAM.items()},
        'region': {v: k for k, v in DICT_REGION.items()},
        'foreign': {v: k for k, v in DICT_FOREIGN.items()},
        'scholarship.type': {v: k for k, v in DICT_SCHOLARSHIP.items()},
        'socioeconomic.level': {v: k for k, v in DICT_SOCIOECONOMIC.items()}
    }

    for col, rev_dict in reverse_dicts.items():
        df_resampled[col] = df_resampled[col].map(rev_dict)

    df_resampled.to_csv(output_path, index=False)
    print(f"Dataset preprocesado guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    # Define la ruta relativa asumiendo que se ejecuta desde la raíz del repositorio
    preprocess_data(input_path="../data/dataset on student dropout.csv", 
                    output_path="../data/dataset_preprocesado.csv")