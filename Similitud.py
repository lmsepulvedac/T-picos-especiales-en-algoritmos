# -*- coding: UTF-8 -*-
import argparse #librería para leer argumentos de la consola
import json #librería para leer archivos con extensión json
import numpy as np

#********************* Función para leer argumentos*************
def lector_argumentos():
    lector = argparse.ArgumentParser(description = 'Cálculo del score de similitud')
    lector.add_argument('--elele1', dest = 'elele1', required = False, help = 'El elele1 es el primer usuario')
    lector.add_argument('--elele2', dest = 'elele2', required = True, help = 'El elele2 es el segundo usuario')
    lector.add_argument('--elele-score', dest = 'elele_score', required = True, choices = ['Euclidiano', 'Pearson'], help = 'Medidas de similitud')
    return lector


#******************Función para generar el score Euclidiano*************
def score_euclidiano(datos, elele1, elele2):
    #comprobamos que los usuarios estén en la base de datos
    if elele1 not in datos:
        raise TypeError('No puedo encontrar a ese elele1 en la base de datos')
    if elele2 not in datos:
        raise TypeError('No puedo encontrar a ese elele2 en la base de datos')
    #revisamos cuántas películas tenemos en común
    peliculas_comunes = {} #declaramos un diccionario vacío
    for item in datos[elele1]:
        if item in datos[elele2]:
            peliculas_comunes[item] = 1
    if len(peliculas_comunes) == 0:
        return 0
    #calculamos el score euclidiano si  tenemos películas en común
    cuadrado_diferencias = [] #inicializamos una lista
    for item in datos[elele1]:
        if item in datos[elele2]:
            cuadrado_diferencias.append(np.square(datos[elele1][item] - datos[elele2][item]))
    score = 1/(1 + np.sqrt(np.sum(cuadrado_diferencias)))
    return score

#***********************Función para generar el score de Pearson
def score_pearson(datos, elele1, elele2):
    #comprobamos que los usuarios estén en la base de datos
    if elele1 not in datos:
        raise TypeError('No puedo encontrar a ese elele1 en la base de datos')
    if elele2 not in datos:
        raise TypeError('No puedo encontrar a ese elele2 en la base de datos')
    #revisamos cuántas películas tenemos en común
    peliculas_comunes = {} #declaramos un diccionario vacío
    for item in datos[elele1]:
        if item in datos[elele2]:
            peliculas_comunes[item] = 1
    if len(peliculas_comunes) == 0:
        return 0
    #calculamos la suma de los rating de todas las películas en común
    suma_usuario1 = np.sum([datos[elele1][item] for item in peliculas_comunes])
    suma_usuario2 = np.sum([datos[elele2][item] for item in peliculas_comunes])
    #calculamos la suma de los cuadrados de las películas en común
    cuadrados_usuario1 = np.sum([np.square(datos[elele1][item]) for item in peliculas_comunes])
    cuadrados_usuario2 = np.sum([np.square(datos[elele2][item]) for item in peliculas_comunes])
    #calculamos la suma de los productos de las películas en común
    suma_productos = np.sum([datos[elele1][item]*datos[elele2][item] for item in peliculas_comunes])
    #calculamos los índices de Pearson
    Sxy = suma_productos - (suma_usuario1*suma_usuario2/len(peliculas_comunes))
    Sxx = cuadrados_usuario1 - (np.square(suma_usuario1)/len(peliculas_comunes))
    Syy = cuadrados_usuario2 - (np.square(suma_usuario2)/len(peliculas_comunes))
    # si no hay desviación el score es cero
    if Sxx*Syy == 0:
        return 0
    #retornamos el score de Pearson
    return Sxy/np.sqrt(Sxx*Syy)



#**********************Generamos el Main***************************
if __name__ == '__main__':
    argumentos = lector_argumentos().parse_args()
    elele1 = argumentos.elele1
    elele2 = argumentos.elele2
    tipo_score = argumentos.elele_score
    #cargamos los datos
    nombre_archivo = 'ratings.json'
    #abrimos el archivo
    with open(nombre_archivo, 'r') as f:
        datos = json.loads(f.read())
    #calculamos la medida de similitud
    if tipo_score == 'Euclidiano':
        print('\nEl score Euclidiano es: ')
        print(score_euclidiano(datos,elele1,elele2))
    else:
        print('\nEl score de Pearson es:')
        print(score_pearson(datos,elele1,elele2))
    
    










