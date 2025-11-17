import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import openslide
import plotly.express as px
from models import MILModel

# ======================
# CONFIGURACIÓN GLOBAL
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ajusta según tus clases reales
id2label = {0: "CE", 1: "LAA"}
NUM_CLASSES = len(id2label)

BASE_MODELS = Path("models")
BASE_METRICS = Path("metrics")


# ======================
# MODELOS
# ======================


def build_cnn_model(num_classes=NUM_CLASSES):

    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def build_mil_model(num_classes=NUM_CLASSES):
  
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


@st.cache_resource
def load_cnn_model():
    weights_path = BASE_MODELS / "cnn_strip_efficientnet_b0.pth"
    model = build_cnn_model()
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def load_mil_model():
    model = MILModel(
        emb_dim=512,
        attn_hidden=256,
        n_classes=2,
        encoder_pretrained=False  # ya entrenado, no queremos bajar pesos de nuevo
    )
    
    weights_path = BASE_MODELS / "mil_strip.pth"

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


# ======================
# PREPROCESAMIENTO
# ======================

def read_image_any(path, level=-1):
    """
    Lee una imagen .tif (WSI) con OpenSlide O una imagen normal (.jpg, .png)
    y devuelve un PIL.Image en RGB.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        # WSI con OpenSlide
        slide = openslide.OpenSlide(str(path))
        if level == -1:
            level = slide.level_count - 1
        w, h = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
        slide.close()
        return img
    else:
        # Imagen normal (jpg, png, etc.)
        img = Image.open(path).convert("RGB")
        return img




def center_crop(img: Image.Image, crop: int):
    w, h = img.size
    cw, ch = min(crop, w), min(crop, h)
    x = (w - cw) // 2
    y = (h - ch) // 2
    return img.crop((x, y, x + cw, y + ch))


def get_tfms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def tensor_from_pil(img_pil, img_size=224):
    tfm = get_tfms(img_size)
    return tfm(img_pil).unsqueeze(0).to(DEVICE)


def make_mil_tiles(thumb: Image.Image, crop_size: int, n_tiles: int, img_size=224):
    """
    Hace n_tiles crops aleatorios (o centrales) del thumbnail
    y los prepara como batch para el modelo MIL.
    """
    tfm = get_tfms(img_size)
    tiles = []
    w, h = thumb.size

    for i in range(n_tiles):
        # aquí podrías usar random crops para variar
        crop = center_crop(thumb, crop_size)
        tiles.append(tfm(crop).unsqueeze(0))

    batch = torch.cat(tiles, dim=0)  # [N,3,H,W]
    return batch.to(DEVICE)


# ======================
# MÉTRICAS (CURVAS)
# ======================

def plot_history(file_path: Path, title_prefix: str):
    if not file_path.exists():
        st.info(f"No se encontró {file_path.name}.")
        return

    hist = pd.read_csv(file_path)

    if {"epoch", "train_loss", "val_loss"}.issubset(hist.columns):
        fig_loss = px.line(hist, x="epoch",
                           y=["train_loss", "val_loss"],
                           title=f"{title_prefix} - Pérdida")
        st.plotly_chart(fig_loss, use_container_width=True)

    if {"epoch", "train_acc", "val_acc"}.issubset(hist.columns):
        fig_acc = px.line(hist, x="epoch",
                          y=["train_acc", "val_acc"],
                          title=f"{title_prefix} - Accuracy")
        st.plotly_chart(fig_acc, use_container_width=True)

    if {"epoch", "train_f1", "val_f1"}.issubset(hist.columns):
        fig_f1 = px.line(hist, x="epoch",
                         y=["train_f1", "val_f1"],
                         title=f"{title_prefix} - F1")
        st.plotly_chart(fig_f1, use_container_width=True)


# ======================
# UI STREAMLIT
# ======================

st.set_page_config(page_title="STRIP AI Dashboard", layout="wide")

st.title("STRIP AI – Dashboard de Clasificación de Etiología")
st.markdown(
    "Demostración interactiva con dos modelos: "
    "**CNN (ResNet/EfficientNet)** y **MIL (Multiple Instance Learning)**."
)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Configuración del experimento")

    model_choice = st.selectbox(
        "Modelo a usar",
        ["CNN (imagen única)", "MIL (bolsa de tiles)"]
    )

    img_size = st.slider("Tamaño de entrada (px)", 224, 512, 224, step=32)
    crop_size = st.slider("Tamaño de crop sobre thumbnail", 256, 2048, 512, step=128)

    if model_choice.startswith("MIL"):
        n_tiles = st.slider("Número de tiles (MIL)", 2, 16, 6, step=2)
    else:
        n_tiles = 1  # no se usa para CNN

    uploaded = st.file_uploader("Sube una lámina", type=["tif", "tiff", "jpg", "jpeg", "png"])


with col_right:
    st.subheader("Curvas de entrenamiento")

    tab1, tab2 = st.tabs(["CNN", "MIL"])

    with tab1:
        plot_history(BASE_METRICS / "history_cnn.csv", "CNN")

    with tab2:
        plot_history(BASE_METRICS / "history_mil.csv", "MIL")


st.markdown("---")

# ======================
# INFERENCIA
# ======================

if uploaded is not None:
    tmp_path = Path("/tmp") / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    thumb = read_image_any(tmp_path)
    st.image(thumb, caption="Thumbnail del slide (nivel bajo)", use_container_width=True)

    if model_choice.startswith("CNN"):
        st.subheader("Predicción con CNN")
        model = load_cnn_model()

        crop = center_crop(thumb, crop_size)
        st.image(crop, caption=f"Crop central ({crop_size}px)", width=300)

        x = tensor_from_pil(crop, img_size=img_size)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

        pred_id = int(np.argmax(probs))
        pred_label = id2label[pred_id]

        st.success(f"Predicción CNN: **{pred_label}**")
        st.bar_chart(pd.Series(probs, index=[id2label[i] for i in range(NUM_CLASSES)]))

    else:
        st.subheader("Predicción con MIL")
        mil_model = load_mil_model()

        # Generar tiles para la bolsa (bag)
        tiles_batch = make_mil_tiles(
            thumb,
            crop_size=crop_size,
            n_tiles=n_tiles,
            img_size=img_size
        )  # -> tensor [N, 3, H, W]

        tiles_batch = tiles_batch.to(DEVICE)

        bag_sizes = [tiles_batch.size(0)]  

        with torch.no_grad():
            logits_bag, attn_w = mil_model(tiles_batch, bag_sizes)  

            probs = torch.softmax(logits_bag, dim=1).cpu().numpy().ravel()

        pred_id = int(np.argmax(probs))
        pred_label = id2label[pred_id]

        st.success(f"Predicción MIL (bolsa de {n_tiles} tiles): **{pred_label}**")
        st.bar_chart(pd.Series(probs, index=[id2label[i] for i in range(NUM_CLASSES)]))

else:
    st.info("Sube una lámina .tif para ejecutar el modelo.")
