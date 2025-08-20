from ultralytics import RTDETR
import cv2
import matplotlib.pyplot as plt


model = RTDETR("runs/detect/train_rtdetr_x/weights/best.pt")

# 读取图像
image_path = "test1.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 执行对象检测
results = model.predict(img_rgb)

# 获取检测结果
detections = results[0].boxes  # 获取Boxes对象

# 在图像上绘制检测框和标签
if detections is not None:
    for box in detections:
        # 获取xyxy坐标
        x1, y1, x2, y2 = box.xyxy[0]
        # 获取置信度
        score = box.conf[0]
        # 获取类别
        cls = box.cls[0]
        if score > 0.5:  # 设置置信度阈值
            label = f"{model.model.names[int(cls)]}: {score:.2f}"
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果图像
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 保存结果图像
output_path = "test1_result.jpg"
cv2.imwrite(output_path, img)
