---
title: Algoritmo
description: Algoritmo de umbralización multinivel automático
author: Brandon Marquez Salazar y Lang Lovdog Inu Oókami
categories: Tareas
created: 2025-11-25T22:19:54-0600
updated: 2025-11-25T22:51:23-0600
version: 1.1.1
---


# Este archivo muestra los algoritmos propuestos para la solución de la tarea 3


La tarea 3 pide que se realice un umbralizado multinivel automático, este umbralizado
requiere de 4 pasos importantes:

1. Creación del histograma de la imagen
2. Reconocimiento de máximos del histograma
3. Basado en los máximos, 'calcular' los umbrales
4. Umbralización multinivel basado en los umbrales calculados


## Algoritmo de histograma


Este algoritmo supone el caso más simple, en el que una imagen está en escala de grises y,
por lo tanto es de un solo canal, con valores entre 0 y 255.

Supóngase una matriz $A \in \mathbb{Z}_{\leq 255}^{m\times n}$ que representa la imagen a
procesar.

Ahora, defínase una vector $H \in \mathbb{Z}_{\leq 255}^{256}$, siendo el histograma de $A$ 
definido como:

$$
H = \left[\sum_{j=0}^{n-1} \sum_{i=0}^{m-1} \phi(A_{ij})\right]_{l=0}^{255}
$$

, donde

$$
\phi = \begin{cases}
  1 & \text{ si } l = A_{ij} \\
  0 & \text{ si } l \neq A_{ij}
\end{cases}
$$

y $l \in \mathbb{Z}_{\leq 255}$, corresponde al índice de los elementos del vector
