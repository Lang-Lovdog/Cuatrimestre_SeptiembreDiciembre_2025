---
title: Transcripción del diario, capítulo del experimento 3 de la clase Inferencia Estadística
description: author: categories: created: 2025-10-23T22:47:58-0600
updated: 2025-10-24T00:08:06-0600
version: 1.1.1
---

Para hoy, no solo anotaré lo del día sino lo que respecta al trabajo de
mañana. Es que necesito mi mente aclarada.

Bien, el trabajo es sobre clasificación. Tenermos un dataset que muestra
la categorización de películas.

Con DeepSeek estuve viendo un análisis de lo que forman estos vectores de
características

También se graficaron las correlaciones que hay, para
esto, DeepSeek me comentó qu'había unas posibles columnas objetivo:
`Start_Tech_Oscar` y `Collection`, así que fue un punto interesante de
partida. También es posible ver en las comparaciones que los elementos que
menos aportaban correspondían a 3 de categorías: `Marketing_Expense`,
`Avg_Age_actors` y `Twitter_hastags`. Así que decidí retirarlos de los
datasets, bien, lo más importante es poder obtener un buen clasificador al
final.

Para realizar la selección de modelo, hice diferentes rejillas como
experimento.

Cada rejilla correspondía a una familia de clasificadores
tomando como referencia un anterior trabajo realizado para la clase de
Reconocimiento de patrones, del mismo modo tomé un framework (trabajo en
progreso) que se inició como base común y producto debido a proyectos
realizados en las materias de Cómputo Evolutivo y Reconocimiento de
Patrones.

Las familias consideradas fueron las siguientes:


- KNN
    - Coarse
    - Cosine
    - Fine
    - Medium
    - Weighed

- Logistic Regression
    - Normal
    - ElasticNet
    - L1

- Neural Networks
    - MLP de una capa
    - MLP de doble capa
    - Perceptrón

- Probabilístico
    - Bayes Naïve Gaussian
    - LDA

- Árboles
    - Árbol de decisión
    - Random Forest
    - Extra Trees

- Support Vector Machines
    - Cúbico
    - Lineal
    - Cuadrático
    - Rbf

Queda comentar algo bastante relevante pues fueron al rededor de 50
modelos y es que se hizo también una variación de la base de datos para
poder decidir diferentes tratamientos. Cada tratamiento se corresponde
della siguiente forma:

- isolation forest
- quantile
- robust scaling
- unclean
- winsorize

«unclean»
implica que'l dataset no recibió tratamiento alguno de
outliers

Cabe mencionar también que hubo otra forma de variacióön que es la
siguiente:

Una vez identificados los elementos que menor correlación aportaban, se
decide retirarlos [La Rejilla](#la-rejilla) pero, también por cuestiones de
curiosidad, permitir unas insntancias del experimento con dichos
elementos, por lo tanto, lo más complejo llega al hacer los entrenamientos
y las pruebas.


Se sobreentiende que cada modelo tiene su propia combinación de
características y que cada familia formará parte de una sola rejilla, por
lo tanto, en total se tienen:

- 5 Tratamientos de outliers
- 6 Familias de clasificaciones
- 2 Versiones del dataset

Por lo que, en total, siendo que de cada familia únicamente se obtendrá un
modelo, se llega a la cantidad de 60 modelos finales de clasificación.


Fue una cantidad muy grande de clasificacdores, por lo que me vi en la
necesidad d'estructurar los elementos en un json que luego un script
utilizara para realizar las pruebas.

[^1]


[^1]: Es un buen aliado

Una vez realizadas las pruebas procedo a entregar mis resultados.

Por reglamento debo entregar únicamente los de un modelo.

- [ ] Punto importante: No puedo entregar al mejor candidato, pues
     sospecho de su desempeño.
    - Debo revisar algún detalle nell código.

Una de las razones por las que desconfío de los resultados es la presencia
de valores muy altos que casi no corresponden al comportamiento de sus
curvas. Tampoco corresponden a los valores d'entrenamiento (oscilaba entre
0.5 y 0.64).

Guardo estos resultados en un directorio aparte y recurro a no operar
outliers, veamos cómo sale el resultado.

Al retirar el preprocesamiento de outliers permitió cambiar el
comportamiento del f1 score, pero los valores varían de 0 a 0.82.

En fin, creo que lo mejor será revisar las listas y enviar el más
confiable. y más optimista.

Es hora de comenzar a hacer algo más sencillo, luego redacto.

Se me envían los resultados de los clasificadores de mi compañera con
quien será la comparación de rendimientos. Se decidió realizar la
comparación de los dos clasificadores en común que se tiene. Se procederá
con una prueba T. Sin embargo, primero se debrá realizar una prueba
Saphiro de normalidad, para corroborar que el f1score tiene una
distribución normal (desgraciadamente mis resultados difícilmente podrían
ser considerados como tal).

Aún dudo de la veracidad de mis resultados.
