from ultralytics import YOLO
import torch

def main():
    base_model = "models/yolov8n.pt"
    data_cfg = "configs/signatures_detect.yaml"

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Entrenando en {device}")

    model = YOLO(base_model)
    model.train(
        data=data_cfg,
        epochs=100,
        imgsz=640,
        batch=16,
        project="detection",
        name="yolo_signature",
        exist_ok=True,
        device=device,
    )

if __name__ == "__main__":
    main()
