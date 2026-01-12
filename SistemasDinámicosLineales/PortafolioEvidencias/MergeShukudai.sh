#!/bin/bash

## Lista de imágenes
imagenes=(
  "./Escáner_20251214.png"
  "./Escáner_20251214_9 (2).png"
  "./Escáner_20251214_2.png"
  "./Escáner_20251214_8 (2).png"
  "./Escáner_20251214_3.png"
  "./Escáner_20251214_7 (2).png"
  "./Escáner_20251214_4.png"
  "./Escáner_20251214_6 (2).png"
  "./Escáner_20251214_5.png"
  "./Escáner_20251214_4 (2).png"
  "./Escáner_20251214_7.png"
  "./Escáner_20251214_3 (2).png"
  "./Escáner_20251214_8.png"
  "./Escáner_20251214_2 (2).png"
  "./Escáner_20251214_9.png"
  "./Escáner_20251214_10.png"
  "./Escáner_20251214 (2).png"
)

## Crear un pdf con todas las imágenes
convert "${imagenes[@]}"  "BrandonMarquezSalazar-PortafoioEvidencias.pdf"
