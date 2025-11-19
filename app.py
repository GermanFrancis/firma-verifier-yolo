import os
import glob

import fitz
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

import gdown

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T

YOLO_MODEL_PATH = "models/best.pt"
VGG_CKPT_PATH = "models/vgg_finetuned_classifier.pt"

#GDRIVE MODELS DOWNLOAD
YOLO_GDRIVE_ID = "1oRTDQfd9WeMqnwMSCgaCVgmShrEFbVvE"
VGG_GDRIVE_ID = "1szSLARkG0UzOtQ1NscNS86M-k_fGPdqE"

CONF_THRES = 0.70

REF_DIR = "data/signatures"

EMB_IMGSZ = 224

THRESH_SIM = 0.76

@st.cache_resource
def load_yolo_model():
    ensure_file(YOLO_MODEL_PATH, YOLO_GDRIVE_ID)

    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(f"No se encontró el modelo YOLO en: {YOLO_MODEL_PATH}")
        st.stop()
    model = YOLO(YOLO_MODEL_PATH)
    return model


@st.cache_resource
def load_vgg_feature_extractor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_file(VGG_CKPT_PATH, VGG_GDRIVE_ID)

    checkpoint = torch.load(VGG_CKPT_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    num_classes = len(class_names)
    in_features = vgg.classifier[-1].in_features
    vgg.classifier[-1] = nn.Linear(in_features, num_classes)
    vgg.load_state_dict(checkpoint["model_state_dict"])
    vgg.to(device)
    vgg.eval()

    feature_extractor = nn.Sequential(
        vgg.features,
        vgg.avgpool,
        nn.Flatten(),
        nn.Sequential(*list(vgg.classifier.children())[:-1])
    ).to(device)
    feature_extractor.eval()

    return feature_extractor, class_names, device

@st.cache_resource
def get_vgg_transform():

    return T.Compose([
        T.Resize((EMB_IMGSZ, EMB_IMGSZ)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def compute_vgg_embedding(img_pil, feature_extractor, device, transform):

    img_t = transform(img_pil).unsqueeze(0).to(device)  # (1,3,224,224)
    with torch.no_grad():
        feat = feature_extractor(img_t)  # (1,4096)
    feat = F.normalize(feat, p=2, dim=1)  # normaliza para coseno
    return feat.squeeze(0).cpu()          # (4096,)


def load_image_from_upload(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")
    return image


@st.cache_resource
def load_reference_embeddings_vgg():

    feature_extractor, class_names, device = load_vgg_feature_extractor()
    transform = get_vgg_transform()

    ref_index = {}

    if not os.path.exists(REF_DIR):
        st.warning(f"No se encontró la carpeta de referencias: {REF_DIR}")
        return ref_index

    for dni_path in glob.glob(os.path.join(REF_DIR, "*")):
        if not os.path.isdir(dni_path):
            continue
        dni = os.path.basename(dni_path)
        embs = []
        paths = []
        for img_path in glob.glob(os.path.join(dni_path, "*")):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            emb = compute_vgg_embedding(img, feature_extractor, device, transform)
            embs.append(emb)
            paths.append(img_path)

        if embs:
            ref_index[dni] = {
                "embs": embs,
                "paths": paths,
            }

    if not ref_index:
        st.warning("No se encontraron referencias en data/signatures.")
    return ref_index


def vgg_cosine_match_score(query_img_pil, ref_index, feature_extractor, device, transform):

    if not ref_index:
        return None, 0.0, {}

    q_emb = compute_vgg_embedding(query_img_pil, feature_extractor, device, transform)  # (D,)

    best_dni, best_score = None, -1.0
    per_dni_scores = {}

    for dni, info in ref_index.items():
        embs = info["embs"]  # lista de tensores (D,)
        if not embs:
            continue

        ref_mat = torch.stack(embs, dim=0)  # (N_ref, D)
        q_vec = q_emb.unsqueeze(0)          # (1, D)

        sims = F.cosine_similarity(q_vec, ref_mat, dim=1)  # (N_ref,)
        max_sim = sims.max().item()
        per_dni_scores[dni] = float(max_sim)

        if max_sim > best_score:
            best_score = float(max_sim)
            best_dni = dni

    if best_score < 0:
        best_score = 0.0
        best_dni = None

    return best_dni, best_score, per_dni_scores

def pdf_to_images_pymupdf(pdf_bytes, dpi=200):

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    zoom = dpi / 72

    for page in pdf:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # convertir a PIL
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)

    return pages

def ensure_file(path, gdrive_id: str | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return

    if gdrive_id is None:
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")

    st.write(f"Descargando modelo desde Google Drive a: {path} ...")

    gdown.download(id=gdrive_id, output=path, quiet=False)


def main():

    st.set_page_config(page_title="Verificador de firmas (YOLO + VGG16)", layout="wide")
    st.title("Verificador de firmas (YOLO + VGG16 Embeddings)")

    st.sidebar.header("1. Subir archivo")
    uploaded_file = st.sidebar.file_uploader(
        "Sube un PDF o una imagen",
        type=["pdf", "jpg", "jpeg", "png"],
    )

    original_image = None

    if uploaded_file is not None:
        file_type = uploaded_file.type

        # PDF -> lista de páginas (imágenes)
        if file_type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pages = pdf_to_images_pymupdf(pdf_bytes, dpi=200)
            original_image = pages[0]

        else:
            original_image = load_image_from_upload(uploaded_file)

        st.subheader("2. Documento cargado")
        st.image(original_image, caption="Página / imagen completa", width='stretch')

        if st.button("Detectar firmas y verificar"):
            yolo_model = load_yolo_model()
            feature_extractor, class_names, device = load_vgg_feature_extractor()
            transform = get_vgg_transform()
            ref_index = load_reference_embeddings_vgg()

            img_np = np.array(original_image)  # PIL -> numpy (RGB)
            results = yolo_model(img_np)
            r = results[0]

            st.write("---")
            if r.boxes is None or len(r.boxes) == 0:
                st.warning("YOLO no detectó firmas en el documento.")
            else:
                st.write("### Detecciones YOLO en el documento")
                pred_img = r.plot()
                pred_img = Image.fromarray(pred_img[..., ::-1])  # BGR->RGB
                st.image(pred_img, width='stretch')

                st.write("### Resultados por firma detectada")
                h, w, _ = img_np.shape
                for i, box in enumerate(r.boxes):
                    conf = float(box.conf[0])
                    if conf < CONF_THRES:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Seguridad de rangos
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))

                    crop = original_image.crop((x1, y1, x2, y2))

                    st.write(f"#### Firma #{i+1}")
                    st.image(
                        crop,
                        caption=f"Recorte firma #{i+1} (conf YOLO = {conf:.2f})",
                        width='content'
                    )

                    best_dni, best_score, per_dni_scores = vgg_cosine_match_score(
                        crop, ref_index, feature_extractor, device, transform
                    )

                    if best_dni is not None:
                        st.write(
                            f"- VGG16 → mejor coincidencia: **{best_dni}** "
                            f"(similitud coseno = {best_score:.3f})"
                        )
                        with st.expander("Similitudes por DNI"):
                            st.json({k: round(v, 3) for k, v in per_dni_scores.items()})
                    else:
                        st.write("- VGG16 → no se pudo calcular similitud (no hay referencias).")

                    if best_dni is not None and best_score >= THRESH_SIM:
                        st.success(
                            f"Firma #{i+1} VERIFICADA como perteneciente a **{best_dni}** "
                            f"(similitud = {best_score:.3f})"
                        )
                    else:
                        st.error(
                            f"Firma #{i+1} NO RECONOCIDA O FALSIFICADA "
                            f"(similitud máxima = {best_score:.3f})"
                        )

    else:
        st.info("Sube un PDF desde la barra lateral para comenzar.")


if __name__ == "__main__":
    main()
