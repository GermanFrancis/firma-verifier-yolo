# 🖋️ Verificador de Firmas digitales (YOLO + VGG16)

Sistema completo para **detección y verificación de firmas manuscritas** utilizando:

- **YOLO** → Detecta automáticamente firmas en documentos PDF.
- **VGG16 Fine-Tuned + Cosine Similarity** → Extrae un embedding y compara contra firmas registradas por DNI.

Apto para validación documental, flujos administrativos y detección de firmas falsas.

---

## 📂 Modelos Entrenados (Google Drive)

Los modelos necesarios para ejecutar la aplicación:

🔗 **https://drive.google.com/drive/folders/1-bQbJNJRPXwde4296cWSNUeg8PhfYjk8?usp=sharing**

## 📂 Carpeta Test (Google Drive)

documentos pdf para probar la aplicación:

🔗 **https://drive.google.com/drive/folders/1mFYWGTqMy4i7ytwmkQIeYugnopjLCqKT?usp=sharing**

Incluye:
- `best.pt` — Detector YOLO de firmas  
- `vgg_finetuned_classifier.pt` — Modelo VGG16 finetuneado  
- Carpeta `signatures/` — Firmas de referencia por DNI  

---

## 🚀 Características

- Detección de firmas en PDFs
- Conversión rápida PDF → imagen (PyMuPDF)
- Recorte automático de cada firma detectada
- Embeddings de 4096 dimensiones (VGG16)
- Verificación por similitud coseno
- Interfaz web lista con Streamlit
- Compatible con despliegue en Streamlit Cloud
