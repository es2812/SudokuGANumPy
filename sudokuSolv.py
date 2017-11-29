import numpy as np
import math as math
import re
import pandas as pd

# Resolvedor de sudokus mediante algoritmos genéticos
#
# Esther Cuervo
# 8.11.17

tab = np.genfromtxt('tablero.csv', dtype='int', delimiter=',')

# Constantes. Macros?
tam_tablero = (int)(math.sqrt(tab.size))
tab = tab.reshape(tam_tablero,tam_tablero)
step = (int)(math.sqrt(tam_tablero))
probabilidad_reproduccion = 0.8
probabilidad_mutacion_debil = 0.2
probabilidad_mutacion_fuerte = 0.01
tam_poblacion = 100
# functions cuenta colisiones
def extraercol(array, r):
    return array[r:tam_tablero:step]

def extraerfila(array, r):
    return array[r * step:(r + 1) * step]

def cuentacolisiones(individuo):
    c_columnas = 0
    c_filas = 0
    for k in range(tam_tablero):
        columna = np.apply_along_axis(extraercol, 1, individuo[math.floor(k/step):tam_tablero:step], k%step)
        freqs = np.unique(columna, return_counts=True)[1]
        # las colisiones son la frecuencia con la que ocurre cada valor menos 1
        c_columnas += sum(freqs - 1)
        fila = np.apply_along_axis(extraerfila, 1, individuo[(math.floor(k/step)*step):((math.floor(k/step) + 1)*step)],k%step)
        freqs = np.unique(fila, return_counts=True)[1]
        c_filas += sum(freqs - 1)
    colisiones_a = c_columnas + c_filas
    return 1 if colisiones_a == 0 else 1 / colisiones_a


# aux functions
def dibujarSudoku(individuo):
    sudok = ""
    for k in range(tam_tablero):
        fila = np.apply_along_axis(extraerfila, 1, individuo[(math.floor(k/step)*step):((math.floor(k/step)+1)*step)], k%step).reshape(tam_tablero)
        for i in range(step):
            sudok += np.array_str(fila[i * step:(i + 1) * step])
            sudok += "\t"
        sudok += "\n"
        if k % step == step - 1:
            sudok += "\n"
    return (re.sub('[\[\]]', '', sudok))

def ajustarvalores(perms):
    # reajustar los valores fijos
    for fila in range(tam_tablero):
        for pos in range(tam_tablero):
            if indices_fijos[fila][pos] and perms[fila][pos] != tab[fila][pos]:
                condition = perms[fila] == tab[fila][pos]
                index = np.where(condition)[0]
                perms[fila][pos], perms[fila][index] = perms[fila][index], perms[fila][pos]

def ajustarvaloresgrid(grid,indice_grid):
    for pos in range(tam_tablero):
        if indices_fijos[indice_grid][pos] and grid[pos] != tab[indice_grid][pos]:
            condition = grid == tab[indice_grid][pos]
            index = np.where(condition)[0]
            grid[pos], grid[index] =  grid[index], grid[pos]
# funciones genetico

def roulette(pesos,aleatorio):
    for i in range(tam_poblacion):
        aleatorio -= pesos.iloc[i]
        if(aleatorio <= 0):
            return i
    return tam_poblacion-1

def seleccion(dicc):
    colisiones_sum = np.sum(dicc['fitness'])
    #print(colisiones_sum)
    #normalizado
    dicc['fitness']= dicc['fitness']/colisiones_sum
    padres = np.zeros((tam_poblacion, tam_tablero, tam_tablero),dtype='int')
    dicc.sort_values('fitness',ascending=False,inplace=True)
    #acumulados = np.zeros(tam_poblacion)
    #acumulados[0] = dicc['fitness'].iloc[0]
    #for i in range(1,tam_poblacion):
    #    acumulados[i] = acumulados[i-1]+ dicc['fitness'].iloc[i]
    #print(dicc['fitness'])
    for i in range(tam_poblacion):
        rand = np.random.uniform()
        indice = roulette(dicc['fitness'],rand)
        #print("random: ",rand)
        #print("seleccionado: ",dicc['fitness'].iloc[indice])
        padres[i] = dicc['tablero'].iloc[indice]

    return padres

def recombinacion(padres):
    # sacamos dos padres en cada iteracion
    tam_padres = padres.shape[0]
    index = 0
    nueva_poblacion = np.zeros((tam_poblacion,tam_tablero,tam_tablero),dtype='int64')
    while tam_padres > 0:
        index_padre = np.random.randint(0,tam_padres)
        padre = padres[index_padre]
        padres=np.delete(padres,index_padre,0)
        tam_padres -= 1

        index_madre = np.random.randint(0,tam_padres)
        madre = padres[index_madre]
        padres=np.delete(padres,index_madre,0)
        tam_padres -= 1

        if np.random.random(1) < probabilidad_reproduccion:
            grid_recomb = np.random.randint(0,tam_tablero)
            hijo = np.concatenate((padre[0:grid_recomb],madre[grid_recomb:tam_tablero]))
            hija = np.concatenate((madre[0:grid_recomb],padre[grid_recomb:tam_tablero]))

        else:
            hijo = padre
            hija = madre

        nueva_poblacion[index] = hijo
        index += 1
        nueva_poblacion[index] = hija
        index += 1

    return nueva_poblacion

def mutacion(individuo,tipo):

    if tipo==1:
    #mutacion debil (se intercambian dos posiciones)
        print("mutacion debil")
        grid_intercambiar = np.random.randint(0,tam_tablero)
        posiciones_intercambiar = (np.random.randint(0,tam_tablero),np.random.randint(0,tam_tablero))
        while(indices_fijos[grid_intercambiar][posiciones_intercambiar[0]] or indices_fijos[grid_intercambiar][posiciones_intercambiar[1]]):
            grid_intercambiar = np.random.randint(0,tam_tablero)
            posiciones_intercambiar = (np.random.randint(0,tam_tablero),np.random.randint(0,tam_tablero))
        #print("MUTACION:",individuo[grid_intercambiar])
        individuo[grid_intercambiar][posiciones_intercambiar[0]], individuo[grid_intercambiar][posiciones_intercambiar[1]] =     individuo[grid_intercambiar][posiciones_intercambiar[1]], individuo[grid_intercambiar][posiciones_intercambiar[0]]
        #print("MUTACION:",individuo[grid_intercambiar])
        return individuo
    else:
        if tipo==2:
            print("mutacion fuerte")
            #mutacion fuerte (se rehace un grid entero)
            grid_intercambiar = np.random.randint(0,tam_tablero)
            nueva_grid = np.random.permutation(np.arange(1,tam_tablero+1))
            ajustarvaloresgrid(nueva_grid,grid_intercambiar)
            individuo[grid_intercambiar] = nueva_grid
            return individuo

    #        else:
                #return individuo
        #mutacion media (se intercambian un número aleatorio de posiciones)
    #return individuo

# obtenemos aquellos indices que contienen -1 en el tablero inicial, por lo que son indices que se deben mover, los convertidos a la misma dimensionalidad que los tableros
indices_no_fijos = tab == -1
indices_fijos = tab != -1
indices_no_fijos = indices_no_fijos.reshape(tam_tablero, tam_tablero)
indices_fijos = indices_fijos.reshape(tam_tablero, tam_tablero)

# tablero es un objeto, cuya primera posicion esta distribuida en dos dimensiones, con cada elemento en la segunda dimension representando un cuadrado 3x3 en el sudoku original, y la segunda posicion contiene su fitness function

def main(n_iteraciones):
    # obtenemos permutaciones de todos los elementos desde 1 hasta 9
    sequences = np.tile(np.arange(1, tam_tablero + 1), (tam_tablero, 1))
    poblacion = np.tile(sequences, (tam_poblacion,1))
    poblacion = np.apply_along_axis(np.random.permutation, 1, poblacion).reshape(tam_poblacion, tam_tablero, tam_tablero)
    for j in range(tam_poblacion):
        ajustarvalores(poblacion[j])
    contador = 0

    diccionario = [{'tablero': poblacion[i], 'fitness': cuentacolisiones(poblacion[i])}
                    for i in range(tam_poblacion)]
    poblacion = pd.DataFrame(diccionario)

    while contador < n_iteraciones:
            #se ha encontrado la solución  TODO: con diccionario
            if np.any(poblacion['fitness']==0):
                return poblacion['tablero'].iloc[np.argmin(poblacion['fitness'])]

            j = np.argmax(poblacion['fitness'])
            print("RONDA NÚMERO ",contador,":")
            print("Mejor individuo (nº", j+1, "):")
            print(dibujarSudoku(poblacion['tablero'][j]))
            print("Puntuacion: ", (1/poblacion['fitness'].iloc[j]), "colisiones")

            padres = seleccion(poblacion)
            poblacion = recombinacion(padres)
            while(np.random.uniform() < probabilidad_mutacion_debil):
                individuo_index = np.random.randint(0,tam_poblacion)
                poblacion[individuo_index] = mutacion(poblacion[individuo_index],1)
            while (np.random.uniform()<probabilidad_mutacion_fuerte):
                individuo_index = np.random.randint(0,tam_poblacion)
                poblacion[individuo_index] = mutacion(poblacion[individuo_index],2)

            diccionario = [{'tablero': poblacion[i], 'fitness': cuentacolisiones(poblacion[i])}
                                        for i in range(tam_poblacion)]
            poblacion = pd.DataFrame(diccionario)
            contador += 1

    return poblacion['tablero'][np.argmax(poblacion['fitness'])]

print("Elija tamaño de población (si el número no es par, se usará el próximo número par):")
tam_poblacion = int(input())
if tam_poblacion % 2 != 0:
    tam_poblacion += 1
print("¿Número de iteraciones máximas?")
iter = input()
if(iter == "inf"):
    n_iter = math.inf
else:
    n_iter = int(iter)
print("Comenzando con poblacion",tam_poblacion," y iteraciones ",n_iter,"...")

resultado=main(n_iter)
print("Solución:")
print(dibujarSudoku(resultado))

solucion_real = np.genfromtxt('tableros.csv',dtype='int',delimiter=',').reshape(tam_tablero,tam_tablero)

if(np.all(solucion_real == resultado)):
    print("CORRECTO")
else:
    print("INCORRECTO")
    print("Solución real:")
    print(dibujarSudoku(solucion_real))
