import pandas as pd
from model import KMeansClassifier
import config
from coinex import RequestsClient
import pickle
import os
import sys
import time
import platform
import monitor

def clear_console():
    os_system = platform.system()
    if os_system == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


class Bot():
    def __init__(self):   
        # Cargar datos
        df = pd.read_csv(config.name_dataset)
        
        # Entrenar modelo
        print("\t Generando Modelo")
        self.kmeans_classifier = KMeansClassifier()

        print('entrenando modelo con dataset')
        self.kmeans_classifier.entrenar(df)

        self.operacion_actual = None
        self.tendencia_actual = None
        self.cant_entrenamientos = 1
        self.precio_actual = None
        self.precio_apertura = None
        self.ganancia_pips = 0
        self.balance = 0

        self.data_str = ""

        self.public_key_temp_api = None

        self.save_state()

    

    def iniciar(self):  
        # Predecir tendencia con datos recientes
        client = RequestsClient()
        datos_recientes,precio_actual = client.get_data()

        while True:
            error = None
            s = ""
            try:
                datos_recientes,precio_actual = client.get_data()
                self.precio_actual = precio_actual
                
                if self.data_str != str(datos_recientes):
                    self.kmeans_classifier.entrenar(datos_recientes)
                    print(f"Datos recientes forma: {datos_recientes.shape}")
                    self.tendencia_actual = self.kmeans_classifier.predecir_tendencia(datos_recientes)
                    s = f"[#] Analisis: \t{self.cant_entrenamientos}\n"
                    s += f'[#] Tendencia actual: \t{self.tendencia_actual}\n'
                    s += f'[#] Precio actual: \t{self.precio_actual}\n'

                    if self.operacion_actual == None:
                        if self.tendencia_actual == 'alcista':
                            s = self.__open_long(s)
                        elif self.tendencia_actual == 'bajista':
                            s = self.__open_short(s)
                        elif self.tendencia_actual == 'ninguna':
                            s = self.__keep_position(s)
                    
                    elif self.operacion_actual == 'Long':
                        if self.tendencia_actual == 'alcista':
                            s = self.__keep_position(s)
                        elif self.tendencia_actual == 'bajista':
                            s = self.__close_position(s)
                            s = self.__open_short(s)
                        elif self.tendencia_actual == 'ninguna':
                            s = self.__close_position(s)
                    
                    elif self.operacion_actual == 'Short':
                        if self.tendencia_actual == 'alcista':
                            s = self.__close_position(s)
                            s = self.__open_long(s)
                        elif self.tendencia_actual == 'bajista':
                            s = self.__keep_position(s)
                        elif self.tendencia_actual == 'ninguna':
                            s = self.__close_position(s)
                else:
                    s = f'[#] Tendencia actual: \t{self.tendencia_actual}\n'
                    s += f'[#] Precio actual: \t{self.precio_actual}\n'
                    s = self.__keep_position(s)
            except Exception as e:
                error = str(e)
            
            self.save_state()
            clear_console()
            print(s)
            self.public_key_temp_api = monitor.update_text_code(mensaje=s,public_key_temp_api=self.public_key_temp_api)

            if not error is None:
                print(error)
                tiempo_espera=1
            else:
                tiempo_espera=config.tiempo_espera
            for i in range(tiempo_espera, 0, -1):
                sys.stdout.write("\rTiempo restante: {:02d}:{:02d} ".format(i // 60, i % 60))
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\r" + " " * 50)  # Limpiar la línea después de la cuenta regresiva
            sys.stdout.flush()



            


    def __open_long(self,s):
        self.operacion_actual = 'Long'
        self.precio_apertura = self.precio_actual
        s += '[#] Abriendo Posicion Long\n'
        
        return s


    def __open_short(self,s):
        self.operacion_actual = 'Short'
        self.precio_apertura = self.precio_actual
        s += '[#] Abriendo Posicion Short\n'
        
        return s



    def __close_position(self,s):
        if self.operacion_actual == 'Long':
            s += '[#] Cerrando Posicion Long\n'
        elif self.operacion_actual == 'Short':
            s += '[#] Cerrando Posicion Short\n'
        self.ganancia_pips += self.__calcular_ganancias_pips()
        self.public_key_temp_api = monitor.post_action(valor=self.ganancia_pips,numero_analisis=self.cant_entrenamientos,public_key_temp_api=self.public_key_temp_api)
        self.operacion_actual = None
        self.precio_apertura = None
        return s


    def __keep_position(self,s):
        if self.operacion_actual == 'Long':
            s += '[#] Mantener Posicion Long\n'
            s+= f'[#] Precio de apertura \t{self.precio_apertura}\n'
        elif self.operacion_actual == 'Short':
            s += '[#] Mantener Posicion Short\n'
            s+= f'[#] Precio de apertura \t{self.precio_apertura}\n'
        else:
            s += '[#] NO EJECUTAR MOVIMIENTO.\n'
        return s

    def __calcular_ganancias_pips(self):
        ganancias = 0
        if self.operacion_actual == "Long":
            ganancias = self.precio_actual - self.precio_apertura
        elif self.operacion_actual == "Short":
            ganancias = self.precio_apertura - self.precio_actual
        return ganancias
            

    #LISTO
    def save_state(self):
        with open('00_data.pkl', 'wb') as file:
            pickle.dump(self, file)

    #LISTO
    @staticmethod
    def load_state():
        if os.path.exists('00_data.pkl'):
            with open('00_data.pkl', 'rb') as file:
                return pickle.load(file)
        else:
            return None
