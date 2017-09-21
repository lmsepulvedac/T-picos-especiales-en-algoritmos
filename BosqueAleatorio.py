
# coding: utf-8

import argparse #librería para analizador de argumentos desde la consola

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from Utilidades import visualizar_clasificador


#contruimos el analizador de argumentos
def analizador_argumentos():
    analizador = argparse.ArgumentParser(description = 'Clasificador basado en aprendizaje combinado')
    analizador.add_argument('--tipo-clasificador', dest = 'tipo_clasificador', required = True, choices = ['ba','bea'], help = 'Escriba el tipo de clasificador que desea; puede ser ba o bea')
    return analizador



if __name__=='__main__':
    #ejecutamos la función del análisis de argumento
    argumentos = analizador_argumentos().parse_args()
    tipo_clasificador = argumentos.tipo_clasificador
    #cargamos los datos de entrada
    datos = np.loadtxt('datos_bosques_aleatorios.txt', delimiter =',')
    X, y = datos[:,:-1], datos[:,-1]
    #separamos los datos por clases
    Clase_0 = np.array(X[y==0])
    Clase_1 = np.array(X[y==1])
    Clase_2 = np.array(X[y==2])
    #generamos la figura de los datos de entrada
    fig = plt.figure()
    plt.scatter(Clase_0[:,0], Clase_0[:,1], s=75, facecolors = 'white', edgecolors = 'black', linewidth = 1, marker = 's')
    plt.scatter(Clase_1[:,0], Clase_1[:,1], s=75, facecolors = 'white', edgecolors = 'black', linewidth = 1, marker = 'o')
    plt.scatter(Clase_2[:,0], Clase_2[:,1], s=75, facecolors = 'white', edgecolors = 'black', linewidth = 1, marker = '^')
    plt.title('Datos de entrada')
    fig.savefig('Datos de entrada.png')
    #dividimos la base de datos en entrenamiento y validación
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25, random_state = 5)
    #establecemos los parámetros del clasificador
    #n_estimators identifica el número de árboles de decisión
    parametros = {'n_estimators': 100, 'max_depth': 4,  'random_state': 0}
    if tipo_clasificador == 'ba':
        clasificador = RandomForestClassifier(**parametros)
    else:
        clasificador = ExtraTreesClassifier(**parametros)
    #entrenamos el clasificador
    clasificador.fit(X_train, y_train)
    visualizar_clasificador(clasificador, X_train, y_train, 'Entrenamiento.png')
    #validamos el clasificador
    y_test_pred = clasificador.predict(X_test)
    visualizar_clasificador(clasificador, X_test, y_test_pred, 'Validacion.png')
    #hacemos el reporte de clasificación
    nombre_clases = ['Clase_0', 'Clase_1', 'Clase_2']
    print('\n' + '#'*70)
    print('\nDesempeño del clasificador en el conjunto de entrenamiento')
    print(classification_report(y_train, clasificador.predict(X_train), target_names = nombre_clases))
    print('#'*70 + '\n')
    
    print('\n' + '#'*70)
    print('\nDesempeño del clasificador en el conjunto de validación')
    print(classification_report(y_test, y_test_pred, target_names = nombre_clases))
    print('#'*70 + '\n')
    
    #vamos a medir la confiabilidad de las predicciones, (intervalo de confianza), para ello generamos un pequeño toy set
    toy_set = np.array([[5,5], [3,6], [6,4], [7,2], [4,4], [5,2]])
    print('\nLa medida de confiabilidad es:')
    for datos in toy_set:
        probabilidad = clasificador.predict_proba([datos])[0]
        clase_pred = 'Clase-' + str(np.argmax(probabilidad))
        print('\nDatos: ', datos)
        print('\nProbabilidad: ',probabilidad)
        print('\nClase: ',clase_pred)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



















