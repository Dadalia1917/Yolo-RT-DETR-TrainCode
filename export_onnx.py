from ultralytics import YOLO

model = YOLO("Yolo训练代码/runs/detect/train_v11/weights/best.pt")
model.export(format="onnx", dynamic=True)
