from ultralytics import YOLO

model_path = r"D:\无锡捷普迅智能科技有限公司\叉车\托盘识别\数据处理\Yolo训练代码\runs\detect\train_v11\weights\best.pt"
model = YOLO(model_path)

model.export(format="onnx")
