from data_processor import DataProcessor
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import joblib

class KMeansClassifier:
    def __init__(self, n_clusters=3, random_state=0):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_labels = []

    def __definir_umbrales(self, tasas_cambio):
        percentil_33 = np.percentile(tasas_cambio, 33)
        percentil_66 = np.percentile(tasas_cambio, 66)
        return percentil_33, percentil_66
    
    def entrenar(self, df : pd.DataFrame):
        # Utilizar la clase DataProcessor para preprocesar los datos

        data_processor = DataProcessor(df)
        print('\tcalculando indicadores')
        data_processor.calcular_indicadores()
        print("\tnormalizando datos")
        data_processor.normalizar()
        print("\tcreando ventanas deslizantes")
        ventanas = data_processor.crear_ventanas()
        
        
        # Entrenar el modelo KMeans
        print('[#] Entrenando modelo')
        self.kmeans.fit(ventanas)
        joblib.dump(self.kmeans, 'modelo_kmeans.pkl')
        
        # Definir umbrales óptimos
        self.cluster_centers = self.kmeans.cluster_centers_
        tasas_cambio = self.cluster_centers[:, -1]
        umbral_bajo, umbral_alto = self.__definir_umbrales(tasas_cambio)
        for center in self.cluster_centers:
            tasa_cambio = center[-1]
            if tasa_cambio > umbral_alto:
                self.cluster_labels.append('alcista')
            elif tasa_cambio < umbral_bajo:
                self.cluster_labels.append('bajista')
            else:
                self.cluster_labels.append('ninguna')


    def __etiquetar_tendencias(self, cluster):
        return self.cluster_labels[cluster]
    

    def predecir_tendencia(self, datos_recientes):
        # Cargar el modelo entrenado
        self.kmeans = joblib.load('modelo_kmeans.pkl')

        # Utilizar la clase DataProcessor para preprocesar los datos recientes
        data_processor = DataProcessor(datos_recientes)
        data_processor.calcular_indicadores()
        data_processor.normalizar()
        
        ventana_reciente = data_processor.crear_ventana_reciente()
        
        print(f"Forma de ventana_reciente: {ventana_reciente.shape}")

        # Predecir y etiquetar
        cluster = self.kmeans.predict([ventana_reciente])[0]
        return self.__etiquetar_tendencias(cluster)
