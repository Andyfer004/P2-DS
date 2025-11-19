# Proyecto 2 – Data Science (P2-DS)

Proyecto de la asignatura de Data Science en el que se desarrollan y comparan
dos enfoques de Deep Learning para la clasificación de imágenes
histopatológicas:

- Una **CNN** sobre patches individuales.
- Un enfoque de **Multiple Instance Learning (MIL)** para bolsas de tiles.

En la fase final, ambos modelos se integran en un **dashboard interactivo en
Streamlit**.

---

## Estructura del repositorio

```text
P2-DS/
├── fase1/
│   ├── data/                         # Datos crudos / preprocesados de la fase 1
│   └── reports/
│       ├── GRUPO 7 Presentación Analisis Exploratorio.pdf
│       ├── GRUPO 7 Presentación.pptx
│       ├── informe_eda.pdf           # Informe del análisis exploratorio
│       └── P2_DS.ipynb               # Notebook de EDA
├── fase2/
│   ├── cnn-p2-ds.ipynb               # Notebook de entrenamiento CNN
│   ├── MIL_nb.ipynb                  # Notebook de entrenamiento MIL
│   ├── informe_seleccion_algoritmo.pdf
│   └── presentacion_seleccion_algoritmo.pdf
├── fase3/
│   ├── images/
│   │   ├── 00c058_0_tissue2048.jpg
│   │   ├── 008e5c_0_tissue2048.jpg
│   │   └── 006388_0_tissue2048.jpg   # Ejemplos de patches extraídos
│   ├── metrics/
│   │   ├── history_cnn.csv           # Historial de entrenamiento CNN
│   │   └── history_mil.csv           # Historial de entrenamiento MIL
│   └── models/
│       ├── __init__.py
│       ├── MIL.py                    # Implementación del modelo MIL
│       ├── app.py                    # App principal de Streamlit (dashboard)
│       ├── resize.py                 # Utilidades para extraer y redimensionar tiles
│       ├── cnn_strip_efficientnet_b0.pth
│       ├── cnn_strip.pth
│       ├── mil_strip.pth
│       ├── mil_resnet_model.pth
│       ├── Presentación Fase Final.pdf
│       └── Proyecto_2_Informe Final.pdf
├── .gitignore
└── requirements.txt                  # Dependencias del proyecto
```

---

## Descripción por fases

### Fase 1 – Análisis Exploratorio (EDA)

- Carga y exploración inicial del dataset.
- Análisis de distribución de clases y balanceo.
- Revisión de calidad de imágenes y posibles artefactos.
- Definición del problema de clasificación y de las variables objetivo.

El trabajo de esta fase está documentado en `P2_DS.ipynb` y en los reportes del
directorio `fase1/reports/`.

### Fase 2 – Selección de algoritmo

Se comparan dos enfoques de modelado:

1. **CNN clásica (imagen única)**  
   - Entrenamiento de una CNN basada en un backbone tipo EfficientNet.  
   - Ajuste de la última capa totalmente conectada para predecir la clase de
     la imagen.

2. **Multiple Instance Learning (MIL)**  
   - Extracción de embeddings de tiles con un encoder convolucional.  
   - Mecanismo de atención para combinar los embeddings de cada bolsa
     (bag) y producir una representación global.  
   - Clasificador final sobre dicha representación.

Los detalles de entrenamiento, métricas y selección del modelo se documentan en
`cnn-p2-ds.ipynb`, `MIL_nb.ipynb` y los informes de `fase2/`.

### Fase 3 – Integración y Dashboard

En esta fase se construye el dashboard **Streamlit** que permite:

- Subir una imagen (o lámina) en formatos `.tif`, `.tiff`, `.jpg`, `.jpeg` o
  `.png`.
- Procesar la imagen para obtener tiles o un patch central representativo.
- Ejecutar la predicción con:
  - El modelo **CNN** para un solo patch.
  - El modelo **MIL** para una bolsa de patches.
- Visualizar:
  - La clase predicha y las probabilidades por clase.
  - Gráficas de la historia de entrenamiento de la CNN y del modelo MIL,
    usando los archivos `history_cnn.csv` e `history_mil.csv`.

`resize.py` contiene funciones auxiliares para extraer y redimensionar tiles a
partir de imágenes de alta resolución.

---

## Requerimientos

- Python 3.x
- PyTorch y torchvision
- Streamlit
- OpenSlide y openslide-python (para WSIs en formato .tif)
- pandas, numpy, pillow, plotly

Las versiones recomendadas se detallan en `requirements.txt`.

---

## Instalación

1. Clonar el repositorio:

```bash
git clone <URL_DEL_REPO>
cd P2-DS
```

2. Crear y activar un entorno virtual:

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\\Scripts\\activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Verificar que los archivos de pesos `.pth` estén ubicados en `fase3/models/`:

- `cnn_strip_efficientnet_b0.pth`
- `cnn_strip.pth` (si aplica)
- `mil_strip.pth`
- `mil_resnet_model.pth`

---

## Ejecución del dashboard

Desde el directorio `fase3/models/`:

```bash
cd fase3/models
streamlit run app.py
```

Luego, abrir en el navegador la URL que muestre Streamlit
(`http://localhost:8501`).

Flujo típico de uso:

1. Seleccionar el modelo a utilizar (CNN o MIL).
2. Configurar parámetros básicos (tamaño de entrada, tamaño de tile, etc.).
3. Subir la imagen o lámina.
4. Consultar la predicción y las métricas de entrenamiento.

---

## Trabajo futuro

- Visualización de mapas de atención para el modelo MIL.
- Soporte para más clases o subtipos de la enfermedad.
- Integración de técnicas de interpretabilidad (Grad-CAM, saliency maps).
- Registro de predicciones en un backend para auditoría y trazabilidad.

---

## Autores

Proyecto desarrollado por el **Grupo 7** de la asignatura de Data Science.

- Davis Roldan
- Andy Fuentes
- Gabriel Paz
- Jose Marchena
