# Preparación de Datos - Employee Attrition
# Este script contiene la preparación completa de datos para el modelo de ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Cargar datos ya limpios del EDA"""
    df = pd.read_csv('../Data/csv/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
    
    # Aplicar limpieza básica del EDA
    df_clean = df.copy()
    
    # Eliminar variables sin valor predictivo
    variables_to_remove = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df_clean = df_clean.drop(columns=[col for col in variables_to_remove if col in df_clean.columns])
    
    # Eliminar inconsistencias lógicas
    df_clean = df_clean[df_clean['DistanceFromHome'] > 0]
    df_clean = df_clean[df_clean['Age'] >= 18]
    df_clean = df_clean[df_clean['YearsAtCompany'] <= df_clean['TotalWorkingYears']]
    
    print(f"Dataset después de limpieza: {df_clean.shape[0]} registros, {df_clean.shape[1]} variables")
    return df_clean

def clean_and_impute_data(df):
    """a) Limpieza de datos, eliminar o imputar datos faltantes, outliers, etc."""
    print("\n" + "="*60)
    print("a) LIMPIEZA DE DATOS Y MANEJO DE DATOS FALTANTES")
    print("="*60)
    
    df_clean = df.copy()
    
    # Verificar valores nulos
    null_counts = df_clean.isnull().sum()
    print("Valores nulos por columna:")
    print(null_counts[null_counts > 0])
    
    if null_counts.sum() == 0:
        print("✅ No hay valores nulos en el dataset")
    else:
        # Estrategia de imputación
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Imputar con la mediana para variables numéricas
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"Imputado {col} con mediana: {median_val}")
                else:
                    # Imputar con la moda para variables categóricas
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"Imputado {col} con moda: {mode_val}")
    
    # Manejo de outliers usando IQR (opcional, mantener para análisis)
    print("\nAnálisis de outliers:")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    outlier_info = []
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_percentage = (len(outliers) / len(df_clean)) * 100
        
        outlier_info.append({
            'Variable': col,
            'Outliers': len(outliers),
            'Porcentaje': outlier_percentage
        })
    
    outlier_df = pd.DataFrame(outlier_info)
    print(outlier_df)
    
    print("\nDecisión: Mantener outliers porque representan casos reales importantes")
    
    return df_clean

def transform_to_numerical(df):
    """b) Transformar todas las variables en valores numéricos"""
    print("\n" + "="*60)
    print("b) TRANSFORMACIÓN A VALORES NUMÉRICOS")
    print("="*60)
    
    df_transformed = df.copy()
    
    # Identificar variables categóricas
    categorical_cols = df_transformed.select_dtypes(include=['object']).columns.tolist()
    print(f"Variables categóricas encontradas: {categorical_cols}")
    
    # Estrategia de encoding
    encoding_strategy = {}
    
    for col in categorical_cols:
        unique_values = df_transformed[col].nunique()
        print(f"\n{col}: {unique_values} valores únicos")
        print(f"Valores: {df_transformed[col].unique()}")
        
        if unique_values == 2:
            # Binary encoding para variables binarias
            le = LabelEncoder()
            df_transformed[col] = le.fit_transform(df_transformed[col])
            encoding_strategy[col] = 'Binary Encoding'
            print(f"Aplicado Binary Encoding")
        elif unique_values <= 10:
            # One-hot encoding para variables categóricas con pocas categorías
            dummies = pd.get_dummies(df_transformed[col], prefix=col)
            df_transformed = pd.concat([df_transformed, dummies], axis=1)
            df_transformed.drop(columns=[col], inplace=True)
            encoding_strategy[col] = 'One-Hot Encoding'
            print(f"Aplicado One-Hot Encoding ({len(dummies.columns)} nuevas columnas)")
        else:
            # Label encoding para variables con muchas categorías
            le = LabelEncoder()
            df_transformed[col] = le.fit_transform(df_transformed[col])
            encoding_strategy[col] = 'Label Encoding'
            print(f"Aplicado Label Encoding")
    
    print(f"\nResumen de transformaciones:")
    for col, strategy in encoding_strategy.items():
        print(f"- {col}: {strategy}")
    
    print(f"\nDataset después de transformación: {df_transformed.shape[1]} variables")
    return df_transformed, encoding_strategy

def normalize_data(df):
    """c) Realizar normalización (Scaler)"""
    print("\n" + "="*60)
    print("c) NORMALIZACIÓN DE DATOS")
    print("="*60)
    
    df_normalized = df.copy()
    
    # Identificar variables numéricas (excluyendo la variable objetivo)
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    if 'Attrition' in numeric_cols:
        numeric_cols.remove('Attrition')
    
    print(f"Variables a normalizar: {numeric_cols}")
    
    # Aplicar StandardScaler
    scaler = StandardScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    
    print("✅ Normalización completada usando StandardScaler")
    print("Todas las variables numéricas ahora tienen media=0 y desviación estándar=1")
    
    # Verificar normalización
    print("\nVerificación de normalización:")
    for col in numeric_cols[:5]:  # Mostrar solo las primeras 5
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        print(f"{col}: media={mean_val:.6f}, std={std_val:.6f}")
    
    return df_normalized, scaler

def feature_reduction(df):
    """d) Aplicar algoritmo de reducción de características (PCA)"""
    print("\n" + "="*60)
    print("d) REDUCCIÓN DE CARACTERÍSTICAS")
    print("="*60)
    
    # Separar características y variable objetivo
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    print(f"Variables originales: {X.shape[1]}")
    
    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Calcular varianza explicada acumulada
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Encontrar número de componentes que explican 95% de la varianza
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    print(f"Número de componentes para 95% de varianza: {n_components_95}")
    print(f"Varianza explicada por los primeros {n_components_95} componentes: {cumulative_variance[n_components_95-1]:.3f}")
    
    # Visualizar varianza explicada
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% de varianza')
    plt.axvline(x=n_components_95, color='g', linestyle='--')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada Acumulada')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('preparacion_datos/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Aplicar PCA con número óptimo de componentes
    pca_final = PCA(n_components=n_components_95)
    X_reduced = pca_final.fit_transform(X)
    
    print(f"\nVariables después de PCA: {X_reduced.shape[1]}")
    print(f"Reducción de dimensionalidad: {X.shape[1]} → {X_reduced.shape[1]}")
    
    # Crear DataFrame con componentes principales
    pca_columns = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
    df_pca = pd.DataFrame(X_reduced, columns=pca_columns)
    df_pca['Attrition'] = y.values
    
    return df_pca, pca_final

def feature_selection(df):
    """e) Algoritmos para selección de características"""
    print("\n" + "="*60)
    print("e) SELECCIÓN DE CARACTERÍSTICAS")
    print("="*60)
    
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    print(f"Variables antes de selección: {X.shape[1]}")
    
    # 1. SelectKBest con f_classif
    print("\n1. SelectKBest con f_classif:")
    selector_kbest = SelectKBest(score_func=f_classif, k=15)
    X_kbest = selector_kbest.fit_transform(X, y)
    
    selected_features_kbest = X.columns[selector_kbest.get_support()].tolist()
    print(f"Características seleccionadas: {selected_features_kbest}")
    
    # 2. Random Forest para importancia de características
    print("\n2. Random Forest - Importancia de características:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 características más importantes:")
    print(feature_importance.head(10))
    
    # 3. Lasso para selección de características
    print("\n3. Lasso para selección de características:")
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X, y)
    
    lasso_features = X.columns[lasso.coef_ != 0].tolist()
    print(f"Características seleccionadas por Lasso: {lasso_features}")
    
    # 4. XGBoost para importancia
    print("\n4. XGBoost - Importancia de características:")
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X, y)
    
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 características más importantes (XGBoost):")
    print(xgb_importance.head(10))
    
    # Combinar resultados y seleccionar características finales
    all_selected = set(selected_features_kbest) | set(lasso_features)
    
    # Agregar top características de Random Forest y XGBoost
    top_rf_features = feature_importance.head(10)['feature'].tolist()
    top_xgb_features = xgb_importance.head(10)['feature'].tolist()
    
    final_features = list(set(list(all_selected) + top_rf_features[:5] + top_xgb_features[:5]))
    
    print(f"\nCaracterísticas finales seleccionadas ({len(final_features)}):")
    print(final_features)
    
    # Crear dataset con características seleccionadas
    df_selected = df[final_features + ['Attrition']].copy()
    
    return df_selected, final_features

def split_datasets(df):
    """f) y g) Separar conjuntos de datos"""
    print("\n" + "="*60)
    print("f) y g) DIVISIÓN DE CONJUNTOS DE DATOS")
    print("="*60)
    
    # Separar características y variable objetivo
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    print(f"Dataset completo: {len(df)} registros")
    print(f"Variables: {X.shape[1]}")
    
    # Primera división: Train-Test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nPrimera división (Train-Test 80-20):")
    print(f"Train: {len(X_train)} registros")
    print(f"Test: {len(X_test)} registros")
    
    # Segunda división: Train-Validation (80-20 del train anterior)
    X_train2, X_validation, y_train2, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nSegunda división (Train-Validation 80-20 del train):")
    print(f"Train2: {len(X_train2)} registros")
    print(f"Validation: {len(X_validation)} registros")
    
    # Verificar distribución de clases
    print(f"\nDistribución de clases:")
    print(f"Train2 - No: {(y_train2 == 'No').sum()}, Yes: {(y_train2 == 'Yes').sum()}")
    print(f"Validation - No: {(y_validation == 'No').sum()}, Yes: {(y_validation == 'Yes').sum()}")
    print(f"Test - No: {(y_test == 'No').sum()}, Yes: {(y_test == 'Yes').sum()}")
    
    return X_train2, X_validation, X_test, y_train2, y_validation, y_test

def save_datasets(X_train2, X_validation, X_test, y_train2, y_validation, y_test):
    """h) Guardar 6 archivos CSV"""
    print("\n" + "="*60)
    print("h) GUARDAR ARCHIVOS CSV")
    print("="*60)
    
    # Crear directorio si no existe
    import os
    os.makedirs('../Data/csv', exist_ok=True)
    
    # Guardar archivos
    datasets = {
        'TrainX.csv': X_train2,
        'TrainY.csv': y_train2,
        'ValidationX.csv': X_validation,
        'ValidationY.csv': y_validation,
        'TestX.csv': X_test,
        'TestY.csv': y_test
    }
    
    for filename, data in datasets.items():
        filepath = f'../Data/csv/{filename}'
        data.to_csv(filepath, index=False)
        print(f"✅ Guardado: {filepath} ({len(data)} registros)")
    
    print(f"\n✅ Todos los archivos guardados en ../Data/csv/")
    print("Archivos creados:")
    print("- TrainX.csv: Características de entrenamiento")
    print("- TrainY.csv: Variable objetivo de entrenamiento")
    print("- ValidationX.csv: Características de validación")
    print("- ValidationY.csv: Variable objetivo de validación")
    print("- TestX.csv: Características de prueba")
    print("- TestY.csv: Variable objetivo de prueba")

def main():
    """Función principal de preparación de datos"""
    print("INICIANDO PREPARACIÓN DE DATOS")
    print("="*60)
    
    # 1. Cargar datos limpios
    df = load_cleaned_data()
    
    # 2. Limpieza adicional e imputación
    df_clean = clean_and_impute_data(df)
    
    # 3. Transformación a valores numéricos
    df_transformed, encoding_strategy = transform_to_numerical(df_clean)
    
    # 4. Normalización
    df_normalized, scaler = normalize_data(df_transformed)
    
    # 5. Reducción de características (PCA)
    df_pca, pca_model = feature_reduction(df_normalized)
    
    # 6. Selección de características
    df_final, selected_features = feature_selection(df_normalized)
    
    # 7. División de datasets
    X_train2, X_validation, X_test, y_train2, y_validation, y_test = split_datasets(df_final)
    
    # 8. Guardar archivos
    save_datasets(X_train2, X_validation, X_test, y_train2, y_validation, y_test)
    
    print("\n" + "="*60)
    print("PREPARACIÓN DE DATOS COMPLETADA")
    print("="*60)
    print("\nResumen del proceso:")
    print("1. ✅ Limpieza de datos y manejo de valores faltantes")
    print("2. ✅ Transformación a valores numéricos")
    print("3. ✅ Normalización con StandardScaler")
    print("4. ✅ Reducción de características con PCA")
    print("5. ✅ Selección de características con múltiples algoritmos")
    print("6. ✅ División Train-Test-Validation")
    print("7. ✅ Archivos CSV guardados")
    print("\nArchivos listos para entrenamiento de modelos de ML")

if __name__ == "__main__":
    main()
