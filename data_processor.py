import numpy as np
import pandas as pd
import config
import sys
class DataProcessor:
    def __init__(self, df : pd.DataFrame):
        self.df = df.drop(columns=['timestamp'], errors='ignore')
        self.df = self.df[['open', 'high', 'low', 'close', 'volume']] # Asegurar orden de columnas
        
    def calcular_indicadores(self):
        # Calcular medias móviles
        self.df['ma_7'] = self.df['close'].rolling(window=7).mean()
        self.df['ma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma_50'] = self.df['close'].rolling(window=50).mean()
        self.df['ma_100'] = self.df['close'].rolling(window=100).mean()
        
        # Eliminar filas con valores nulos
        self.df.dropna(inplace=True)

    def normalizar(self):
        max_high = self.df['high'].max()
        self.df['open'] = self.df['open'] / max_high
        self.df['high'] = self.df['high'] / max_high
        self.df['low'] = self.df['low'] / max_high
        self.df['close'] = self.df['close'] / max_high
        
        self.df['volume'] = self.df['volume'] / self.df['volume'].max()

        self.df['ma_7'] = self.df['ma_7'] / max_high
        self.df['ma_20'] = self.df['ma_20'] / max_high
        self.df['ma_50'] = self.df['ma_50'] / max_high
        self.df['ma_100'] = self.df['ma_100'] / max_high
     

    def crear_ventanas(self, ventana=config.tamanio_ventana):
        data = self.df.values
        close_idx = self.df.columns.get_loc('close')
        ventanas = np.empty((len(self.df) - ventana + 1, ventana * data.shape[1] + 1))  # Matriz de ventanas, +1 para tasa_cambio
        
        for i in range(len(self.df) - ventana + 1):
            ventana_data = data[i:i+ventana]
            tasa_cambio = (ventana_data[-1, close_idx] - ventana_data[0, close_idx]) / ventana_data[0, close_idx]
            ventana_features = np.append(ventana_data.flatten(), tasa_cambio)
            ventanas[i] = ventana_features
            
            sys.stdout.write(f'\r{DataProcessor.barra_progreso(float(i/(len(self.df) - ventana)))}')
            sys.stdout.flush()
        
        return ventanas

    def crear_ventana_reciente(self, ventana=config.tamanio_ventana):
        ventana_df = self.df.iloc[-ventana:]
        tasa_cambio = (ventana_df['close'].iloc[-1] - ventana_df['close'].iloc[0]) / ventana_df['close'].iloc[0]
        ventana_features = np.append(ventana_df.values.flatten(), tasa_cambio)
        return ventana_features

    
    @staticmethod
    def barra_progreso(porcentaje):
        longitud_total = 50
        longitud_progreso = int(longitud_total * porcentaje)
        barra = '[' + '#' * longitud_progreso + '-' * (longitud_total - longitud_progreso) + ']' + f'{round(porcentaje*100,2)}%'
        return barra

