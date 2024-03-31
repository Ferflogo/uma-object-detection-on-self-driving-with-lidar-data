# Detección y seguimiento de objetos mediante aprendizaje profundo para conducción autónoma

## Descripción

El siguiente proyecto es una colaboración entre el estudiante Fernando Flores Gómez del Instituto Tecnológico de Costa Rica y el Laboratorio ICAI de la Universidad de Málaga para optar por el título de Ingeniero en Mecatrónica con el grado académico de Licenciatura. El presente repositorio muestra los pasos para replicar el proyecto de graduación para título: **“Detección y seguimiento de objetos mediante aprendizaje profundo para conducción autónoma”** el cual se puede acceder por medio de la biblioteca virtual del Instituto Tecnológico de Costa Rica.

Este proyecto es basado en el siguiente documento repositorio: [Complex YOLOv4](https://github.com/maudzung/Complex-YOLOv4-Pytorch)

Además, se utilizaron los datos de KITTI los cuales pueden ser descargados desde: [KITTY Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php)

A continuació, se muestran los resultados de dicha implementación.

<figure class="image">
  <p align="center"> <img src="assets/test_bench.gif" alt></p>
  <figcaption align="center">Result Ouput</figcaption>
</figure>

> [!NOTE]  
> En este proyecto es posible generar un conjunto de datos con imágenes LiDAR (imágenes de vista de pájaro generadas a partir de información de nube de puntos), y entrenar cualquier modelo de detección de objetos, con el fin de probar otros modelos como YOLOv8 o robustecer el modelo agregando más objetos.

## Estructura de carpeta

```bash
.
├── assets
|   └── test_bench.gif
├── checkpoints/complex_yolov4
|   └── complex_yolov4_mse_loss.pth
├── complex-env
├── example/dataset/kitti
|   ├── testing
|   |   ├── calib
|   |   ├── image
|   |   └── velodyne
|   └── classes_names.txt
├── src
|   ├── config
|   ├── data_process
|   ├── models
|   └── utils
├── requirements.txt
├── run_model.ipynb
└── README
```

## Especificaciones de sistema

| Componente | Version |
|---|---|
| Python | 3.7.16 |
|PIP | 24.0 |
| Pytorch | 1.7.1 |
| GPU | Nvidia RTX 3060 |
| CUDA | 11.0 |

## Creacion del ambiente de trabajo

Creación de un ambiente python:

```bash
conda create --prefix ./complex-env python=3.7 -y
```

Activar ambiente:

```bash
conda activate ./complex-env
```

Instalación de dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --ignore-installed --no-deps
```

<details close>
<summary>Verificar instalación</summary>
<br>

```bash
python
```
```python
import torch
print(torch.cuda.is_available())
```
Output: 
```python
True
```
</details>

Agregar **Kernel** a Jupyter:

```bash
python -m ipykernel install --user --name=complex-kernel
```

> [!NOTE]  
> También se puede instalar las herramientas de Jupyter y correr el Notebook desde Jupyter.

Correr Notebook:

Se procede a correr el Notebook llamado "run_model.ipynb", en este se presentan los archivos mínimos necesarios para correr la inferencia y obtener los resultados finales.