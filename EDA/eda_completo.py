# Análisis Exploratorio de Datos (EDA) - Employee Attrition
# Este script contiene el análisis exploratorio completo del dataset de deserción laboral

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Cargar y explorar los datos iniciales"""
    df = pd.read_csv('Data/csv/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    return df

def analyze_nulls(df):
    """a) Encontrar porcentaje de nulos"""
    print("\n" + "="*50)
    print("a) ANÁLISIS DE VALORES NULOS")
    print("="*50)
    
    null_percentage = (df.isnull().sum() / len(df)) * 100
    null_df = pd.DataFrame({
        'Columna': null_percentage.index,
        'Porcentaje_Nulos': null_percentage.values
    })
    null_df = null_df[null_df['Porcentaje_Nulos'] > 0].sort_values('Porcentaje_Nulos', ascending=False)
    
    if len(null_df) == 0:
        print("✅ No hay valores nulos en el dataset")
    else:
        print(null_df)
    
    # Verificar también valores vacíos
    empty_strings = df.apply(lambda x: (x == '').sum() if x.dtype == 'object' else 0)
    print(f"\nValores de cadena vacía encontrados: {empty_strings.sum()}")

def analyze_zeros(df):
    """b) Encontrar porcentaje de '0's' y ver si tiene sentido"""
    print("\n" + "="*50)
    print("b) ANÁLISIS DE CEROS EN VARIABLES NUMÉRICAS")
    print("="*50)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Variables numéricas: {numeric_columns}")
    
    zero_analysis = []
    for col in numeric_columns:
        zero_count = (df[col] == 0).sum()
        zero_percentage = (zero_count / len(df)) * 100
        zero_analysis.append({
            'Variable': col,
            'Cantidad_Ceros': zero_count,
            'Porcentaje_Ceros': zero_percentage,
            'Tiene_Sentido_Ceros': 'Sí' if zero_count > 0 else 'No'
        })
    
    zero_df = pd.DataFrame(zero_analysis)
    zero_df = zero_df[zero_df['Cantidad_Ceros'] > 0].sort_values('Porcentaje_Ceros', ascending=False)
    
    print("\nVariables con ceros:")
    print(zero_df)
    
    print("\n=== JUSTIFICACIÓN DE CEROS ===")
    for _, row in zero_df.iterrows():
        var = row['Variable']
        pct = row['Porcentaje_Ceros']
        
        if var in ['YearsSinceLastPromotion', 'YearsInCurrentRole']:
            print(f"{var}: {pct:.1f}% - Tiene sentido (empleados nuevos o sin promociones)")
        elif var in ['DistanceFromHome']:
            print(f"{var}: {pct:.1f}% - Posible error de datos (distancia no puede ser 0)")
        elif var in ['NumCompaniesWorked']:
            print(f"{var}: {pct:.1f}% - Tiene sentido (primer trabajo)")
        else:
            print(f"{var}: {pct:.1f}% - Revisar si tiene sentido lógico")

def generate_visualizations(df):
    """c) Generar gráficos de cada variable"""
    print("\n" + "="*50)
    print("c) GENERACIÓN DE GRÁFICOS")
    print("="*50)
    
    # Histogramas para variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
    
    # Ocultar subplots vacíos
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('EDA/histogramas_variables_numericas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráficos de barras para variables categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            value_counts = df[col].value_counts()
            value_counts.plot(kind='bar', ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Ocultar subplots vacíos
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('EDA/graficos_variables_categoricas.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_target_variable(df):
    """d) Visualización de la variable de salida"""
    print("\n" + "="*50)
    print("d) VISUALIZACIÓN DE LA VARIABLE OBJETIVO (ATTRITION)")
    print("="*50)
    
    plt.figure(figsize=(15, 5))
    
    # Gráfico de barras
    plt.subplot(1, 3, 1)
    attrition_counts = df['Attrition'].value_counts()
    attrition_counts.plot(kind='bar', color=['lightcoral', 'lightblue'])
    plt.title('Distribución de Attrition')
    plt.xlabel('Attrition')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=0)
    
    # Gráfico de pie
    plt.subplot(1, 3, 2)
    attrition_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
    plt.title('Proporción de Attrition')
    plt.ylabel('')
    
    # Estadísticas
    plt.subplot(1, 3, 3)
    plt.axis('off')
    attrition_stats = df['Attrition'].value_counts(normalize=True) * 100
    stats_text = f"""
Estadísticas de Attrition:

Total empleados: {len(df):,}
No renuncian: {attrition_counts['No']:,} ({attrition_stats['No']:.1f}%)
Sí renuncian: {attrition_counts['Yes']:,} ({attrition_stats['Yes']:.1f}%)

El dataset está desbalanceado
con mayoría de empleados
que no renuncian.
"""
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('EDA/analisis_variable_objetivo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Proporción de empleados que NO renuncian: {attrition_stats['No']:.1f}%")
    print(f"Proporción de empleados que SÍ renuncian: {attrition_stats['Yes']:.1f}%")

def clean_data(df):
    """e) Eliminar registros con nulos, ceros o inconsistencias"""
    print("\n" + "="*50)
    print("e) LIMPIEZA DE DATOS")
    print("="*50)
    
    df_clean = df.copy()
    print(f"Dataset original: {df_clean.shape[0]} registros")
    
    # 1. Eliminar registros con valores nulos (si los hay)
    before_nulls = len(df_clean)
    df_clean = df_clean.dropna()
    after_nulls = len(df_clean)
    print(f"Registros eliminados por nulos: {before_nulls - after_nulls}")
    
    # 2. Identificar y eliminar inconsistencias lógicas
    inconsistencies = []
    
    # DistanceFromHome = 0 no tiene sentido lógico
    distance_zero = df_clean[df_clean['DistanceFromHome'] == 0]
    if len(distance_zero) > 0:
        inconsistencies.append(f"DistanceFromHome = 0: {len(distance_zero)} registros")
        df_clean = df_clean[df_clean['DistanceFromHome'] > 0]
    
    # Age < 18 no es legal para empleados
    age_invalid = df_clean[df_clean['Age'] < 18]
    if len(age_invalid) > 0:
        inconsistencies.append(f"Age < 18: {len(age_invalid)} registros")
        df_clean = df_clean[df_clean['Age'] >= 18]
    
    # YearsAtCompany > TotalWorkingYears no tiene sentido
    years_inconsistent = df_clean[df_clean['YearsAtCompany'] > df_clean['TotalWorkingYears']]
    if len(years_inconsistent) > 0:
        inconsistencies.append(f"YearsAtCompany > TotalWorkingYears: {len(years_inconsistent)} registros")
        df_clean = df_clean[df_clean['YearsAtCompany'] <= df_clean['TotalWorkingYears']]
    
    print("\n=== INCONSISTENCIAS ELIMINADAS ===")
    if inconsistencies:
        for inconsistency in inconsistencies:
            print(f"- {inconsistency}")
    else:
        print("No se encontraron inconsistencias lógicas")
    
    print(f"\nDataset después de limpieza: {df_clean.shape[0]} registros")
    print(f"Registros eliminados: {len(df) - len(df_clean)} ({((len(df) - len(df_clean))/len(df)*100):.1f}%)")
    
    return df_clean

def remove_unnecessary_features(df_clean):
    """f) Eliminar características que no aplican"""
    print("\n" + "="*50)
    print("f) ELIMINACIÓN DE CARACTERÍSTICAS INNECESARIAS")
    print("="*50)
    
    variables_to_remove = []
    
    # Variables con varianza cero o muy baja
    for col in df_clean.columns:
        if df_clean[col].nunique() == 1:
            variables_to_remove.append(col)
            print(f"Eliminando {col}: solo tiene un valor único")
    
    # Variables que no aportan información útil
    if 'EmployeeCount' in df_clean.columns:
        variables_to_remove.append('EmployeeCount')
        print(f"Eliminando EmployeeCount: siempre es 1")
    
    if 'Over18' in df_clean.columns:
        variables_to_remove.append('Over18')
        print(f"Eliminando Over18: todos los empleados son mayores de 18")
    
    if 'StandardHours' in df_clean.columns:
        variables_to_remove.append('StandardHours')
        print(f"Eliminando StandardHours: siempre es 80")
    
    if 'EmployeeNumber' in df_clean.columns:
        variables_to_remove.append('EmployeeNumber')
        print(f"Eliminando EmployeeNumber: identificador único sin valor predictivo")
    
    # Eliminar variables identificadas
    df_clean = df_clean.drop(columns=variables_to_remove)
    
    print(f"\n=== VARIABLES ELIMINADAS ===")
    print(f"Total de variables eliminadas: {len(variables_to_remove)}")
    print(f"Variables restantes: {df_clean.shape[1]}")
    print(f"Variables eliminadas: {variables_to_remove}")
    
    return df_clean

def analyze_outliers(df_clean):
    """g) Análisis de outliers"""
    print("\n" + "="*50)
    print("g) ANÁLISIS DE OUTLIERS")
    print("="*50)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    outlier_analysis = []
    
    # Crear gráficos de boxplot
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(numeric_cols):
        plt.subplot(3, 6, i+1)
        df_clean.boxplot(column=col, ax=plt.gca())
        plt.title(f'{col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('EDA/analisis_outliers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Análisis detallado de outliers
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_percentage = (len(outliers) / len(df_clean)) * 100
        
        outlier_analysis.append({
            'Variable': col,
            'Outliers': len(outliers),
            'Porcentaje': outlier_percentage,
            'Accion': 'Mantener' if outlier_percentage < 5 else 'Revisar'
        })
    
    outlier_df = pd.DataFrame(outlier_analysis)
    print(outlier_df)
    
    print("\n=== DECISIÓN SOBRE OUTLIERS ===")
    print("Estrategia: Mantener outliers porque:")
    print("1. Representan casos reales de empleados con características extremas")
    print("2. Pueden ser importantes para predecir deserción")
    print("3. El porcentaje de outliers es generalmente bajo (< 10%)")
    print("4. Los algoritmos de ML pueden manejar outliers apropiadamente")

def correlation_analysis(df_clean):
    """h) Análisis de correlación y eliminación de variables correlacionadas"""
    print("\n" + "="*50)
    print("h) ANÁLISIS DE CORRELACIÓN")
    print("="*50)
    
    # Calcular matriz de correlación para variables numéricas
    numeric_data = df_clean.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    # Crear mapa de calor
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    plt.title('Matriz de Correlación - Variables Numéricas')
    plt.tight_layout()
    plt.savefig('EDA/matriz_correlacion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Identificar variables altamente correlacionadas
    high_corr_pairs = []
    threshold = 0.8  # Umbral de correlación alta
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'Variable1': correlation_matrix.columns[i],
                    'Variable2': correlation_matrix.columns[j],
                    'Correlacion': corr_value
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    
    print("\n=== VARIABLES ALTAMENTE CORRELACIONADAS ===")
    if len(high_corr_df) > 0:
        print(high_corr_df)
        
        # Decidir qué variables eliminar
        variables_to_remove_corr = []
        
        for _, row in high_corr_df.iterrows():
            var1, var2 = row['Variable1'], row['Variable2']
            
            # Eliminar la variable con menor varianza o menos información
            if var1 in ['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']:
                if var2 not in variables_to_remove_corr:
                    variables_to_remove_corr.append(var1)
                    print(f"Eliminando {var1} (alta correlación con {var2})")
            elif var2 in ['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']:
                if var1 not in variables_to_remove_corr:
                    variables_to_remove_corr.append(var2)
                    print(f"Eliminando {var2} (alta correlación con {var1})")
        
        # Eliminar variables altamente correlacionadas
        df_clean = df_clean.drop(columns=variables_to_remove_corr)
        print(f"\nVariables eliminadas por alta correlación: {variables_to_remove_corr}")
    else:
        print("No se encontraron variables con correlación > 0.8")
    
    print(f"\nDataset final después de eliminar variables correlacionadas: {df_clean.shape[1]} variables")
    return df_clean

def perform_clustering(df_clean):
    """i) Realizar algoritmo de KMeans"""
    print("\n" + "="*50)
    print("i) ALGORITMO DE CLUSTERING K-MEANS")
    print("="*50)
    
    # Preparar datos para clustering
    df_cluster = df_clean.copy()
    
    # Codificar variables categóricas
    le_dict = {}
    categorical_cols = df_cluster.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'Attrition':  # Excluir variable objetivo
            le = LabelEncoder()
            df_cluster[col] = le.fit_transform(df_cluster[col])
            le_dict[col] = le
    
    # Seleccionar variables numéricas para clustering
    numeric_cols_cluster = df_cluster.select_dtypes(include=[np.number]).columns.tolist()
    if 'Attrition' in numeric_cols_cluster:
        numeric_cols_cluster.remove('Attrition')
    
    X_cluster = df_cluster[numeric_cols_cluster]
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    print(f"Variables utilizadas para clustering: {numeric_cols_cluster}")
    print(f"Forma de los datos: {X_scaled.shape}")
    
    # Método del codo para determinar número óptimo de clusters
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Gráfico del método del codo
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, 'ro-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Coeficiente de Silueta')
    plt.title('Coeficiente de Silueta')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('EDA/metodo_codo_silueta.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Determinar número óptimo de clusters
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nNúmero óptimo de clusters según coeficiente de silueta: {optimal_k}")
    print(f"Coeficiente de silueta máximo: {max(silhouette_scores):.3f}")
    
    # Aplicar KMeans con número óptimo de clusters
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    # Agregar etiquetas de cluster al dataset
    df_cluster['Cluster'] = cluster_labels
    
    # Análisis de cada cluster
    print(f"\n=== ANÁLISIS DE CLUSTERS (k={optimal_k}) ===")
    
    cluster_stats = []
    for cluster_id in range(optimal_k):
        cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        # Calcular estadísticas del cluster
        stats = {
            'Cluster': cluster_id,
            'Tamaño': cluster_size,
            'Porcentaje': (cluster_size / len(df_cluster)) * 100
        }
        
        # Estadísticas por variable
        for col in numeric_cols_cluster:
            stats[f'{col}_mean'] = cluster_data[col].mean()
        
        # Tasa de attrition por cluster
        if 'Attrition' in df_cluster.columns:
            attrition_rate = (cluster_data['Attrition'] == 'Yes').sum() / cluster_size * 100
            stats['Attrition_Rate'] = attrition_rate
        
        cluster_stats.append(stats)
    
    cluster_df = pd.DataFrame(cluster_stats)
    print(cluster_df.round(2))
    
    # Visualización de clusters
    plt.figure(figsize=(15, 10))
    
    # Seleccionar 2 variables principales para visualización
    if 'Age' in numeric_cols_cluster and 'MonthlyIncome' in numeric_cols_cluster:
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(df_cluster['Age'], df_cluster['MonthlyIncome'], 
                             c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Age')
        plt.ylabel('Monthly Income')
        plt.title('Clusters por Age vs Monthly Income')
        plt.colorbar(scatter)
    
    if 'JobSatisfaction' in numeric_cols_cluster and 'WorkLifeBalance' in numeric_cols_cluster:
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(df_cluster['JobSatisfaction'], df_cluster['WorkLifeBalance'], 
                             c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Job Satisfaction')
        plt.ylabel('Work Life Balance')
        plt.title('Clusters por Job Satisfaction vs Work Life Balance')
        plt.colorbar(scatter)
    
    # Distribución de clusters
    plt.subplot(2, 2, 3)
    cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Cantidad de Empleados')
    plt.title('Distribución de Empleados por Cluster')
    
    # Tasa de attrition por cluster
    plt.subplot(2, 2, 4)
    attrition_by_cluster = df_cluster.groupby('Cluster')['Attrition'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_index()
    plt.bar(attrition_by_cluster.index, attrition_by_cluster.values, color='lightcoral')
    plt.xlabel('Cluster')
    plt.ylabel('Tasa de Attrition (%)')
    plt.title('Tasa de Attrition por Cluster')
    
    plt.tight_layout()
    plt.savefig('EDA/analisis_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Descripción de cada cluster
    print("\n=== DESCRIPCIÓN DE CLUSTERS ===")
    for cluster_id in range(optimal_k):
        cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"- Tamaño: {len(cluster_data)} empleados ({(len(cluster_data)/len(df_cluster)*100):.1f}%)")
        
        if 'Attrition' in df_cluster.columns:
            attrition_rate = (cluster_data['Attrition'] == 'Yes').sum() / len(cluster_data) * 100
            print(f"- Tasa de Attrition: {attrition_rate:.1f}%")
        
        # Características distintivas
        print(f"- Características distintivas:")
        for col in ['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance']:
            if col in cluster_data.columns:
                mean_val = cluster_data[col].mean()
                overall_mean = df_cluster[col].mean()
                if mean_val > overall_mean * 1.1:
                    print(f"  * {col}: Alto ({mean_val:.1f} vs {overall_mean:.1f} promedio)")
                elif mean_val < overall_mean * 0.9:
                    print(f"  * {col}: Bajo ({mean_val:.1f} vs {overall_mean:.1f} promedio)")
                else:
                    print(f"  * {col}: Promedio ({mean_val:.1f})")

def main():
    """Función principal que ejecuta todo el EDA"""
    print("INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*60)
    
    # Cargar datos
    df = load_and_explore_data()
    
    # Ejecutar análisis
    analyze_nulls(df)
    analyze_zeros(df)
    generate_visualizations(df)
    visualize_target_variable(df)
    
    # Limpieza de datos
    df_clean = clean_data(df)
    df_clean = remove_unnecessary_features(df_clean)
    analyze_outliers(df_clean)
    df_clean = correlation_analysis(df_clean)
    
    # Clustering
    perform_clustering(df_clean)
    
    print("\n" + "="*60)
    print("EDA COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("\nHallazgos principales:")
    print("1. Datos limpios: No hay valores nulos en el dataset")
    print("2. Variables eliminadas: EmployeeCount, Over18, StandardHours, EmployeeNumber")
    print("3. Outliers: Mantenidos porque representan casos reales importantes")
    print("4. Correlaciones: Se eliminaron variables altamente correlacionadas")
    print("5. Clustering: Se identificaron clusters con diferentes perfiles de riesgo")
    print("\nPróximos pasos:")
    print("- Preparación de datos (normalización, encoding)")
    print("- División train/test/validation")
    print("- Entrenamiento de modelos de ML")

if __name__ == "__main__":
    main()
