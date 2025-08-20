import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView, QTableWidgetItem, QAbstractItemView
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
from PyQt5.QtCore import QTimer, Qt
import detect_tools as tools
import cv2
import Config2
from UIProgram.QssLoader import QSSLoader
import numpy as np
import torch
import warnings
import logging

logging.basicConfig(filename='debug.log', level=logging.DEBUG)

warnings.filterwarnings('ignore', category=DeprecationWarning)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()

        # Load css rendering effect
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

        self.conf = 0.5
        self.unkonwn_text = '无法识别'

        # Video and camera detection frequency, detect once every 5 frames
        self.detection_frequency = 5

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)
        self.ui.tableWidget.cellClicked.connect(self.on_cell_clicked)
        self.ui.selection_cb.currentIndexChanged.connect(self.on_selection_change)

    def initMain(self):
        self.show_width = 770
        self.show_height = 480

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load detection model
        self.model = YOLO("runs/detect/train_v13/weights/best.pt")
        self.model.fuse()

        # Update video image
        self.timer_camera = QTimer()

        # Table
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.ui.tableWidget.setColumnWidth(0, 100)  # Set column width
        self.ui.tableWidget.setColumnWidth(1, 300)
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # Set table to select entire row
        self.ui.tableWidget.verticalHeader().setVisible(False)  # Hide column header
        self.ui.tableWidget.setAlternatingRowColors(True)  # Alternating row colors

        # Initialize labels
        self.ui.type_lb.setText("")
        self.ui.confidence_lb.setText("")

    def open_img(self):
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            self.cap = None
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path).copy()
        t1 = time.time()
        results = self.model.predict(self.org_img, conf=self.conf)

        # 初始化计数字典（支持所有类别）
        class_counts = {cls_name: 0 for cls_name in self.model.names.values()}

        self.boxes = []
        detections = results[0].boxes
        display_img = self.org_img.copy()

        if detections is not None:
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                score = box.conf[0].item()
                cls = box.cls[0].item()
                if score > self.conf:
                    # 获取英文类别名
                    en_cls = self.model.names[int(cls)]
                    # 更新计数并生成 display_name（保留英文）
                    class_counts[en_cls] += 1
                    display_name = f"{en_cls}{class_counts[en_cls]}"
                    # 添加到 boxes 列表
                    self.boxes.append((x1, y1, x2, y2, display_name, score))
                    # 绘制检测框和标签（保留英文）
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_img, f"{en_cls}: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(f"{take_time_str}")

        # 统计总目标数（所有类别）
        total = sum(class_counts.values())
        self.ui.total_lb.setText(f"{total}")

        # 更新下拉框（保留英文）
        self.ui.selection_cb.clear()
        self.ui.selection_cb.addItem("")
        for box in self.boxes:
            self.ui.selection_cb.addItem(box[4], box)

        # 更新表格
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.org_path)

        # 显示图片
        self.img_width, self.img_height = self.get_resize_size(display_img)
        resize_cvimg = cv2.resize(display_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def on_selection_change(self):
        if not hasattr(self, 'org_img') or self.org_img is None:
            logging.debug("org_img is not defined.")
            return

        selection_text = self.ui.selection_cb.currentText()
        display_img = self.org_img.copy()
        self.ui.type_lb.setText("")  # 类型字段初始化为空
        self.ui.confidence_lb.setText("")
        self.ui.xmin_lb.setText("")
        self.ui.ymin_lb.setText("")
        self.ui.xmax_lb.setText("")
        self.ui.ymax_lb.setText("")

        if selection_text:
            for box in self.boxes:
                x1, y1, x2, y2, label, score = box
                if label == selection_text:
                    # 高亮选中目标（保留英文标签）
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display_img, f"{label}: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    # 设置坐标信息
                    self.ui.xmin_lb.setText(str(x1))
                    self.ui.ymin_lb.setText(str(y1))
                    self.ui.xmax_lb.setText(str(x2))
                    self.ui.ymax_lb.setText(str(y2))
                    self.ui.confidence_lb.setText(f"{score:.2f}")

                    # 提取英文类别名 → 转换为中文
                    en_cls = ''.join(filter(str.isalpha, label))  # 去除计数部分（如 vehicle1 → vehicle）
                    ch_type = Config2.EN_to_CH.get(en_cls, "未知")  # 使用 Config 中的映射
                    self.ui.type_lb.setText(ch_type)
                    break
        else:
            # 重置所有检测框为绿色
            for box in self.boxes:
                x1, y1, x2, y2, label, score = box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"{label}: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示图片
        self.img_width, self.img_height = self.get_resize_size(display_img)
        resize_cvimg = cv2.resize(display_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def detact_batch_imgs(self):
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')
            self.cap = None
        directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        if not directory:
            return

        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.org_path = directory
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        self.boxes_dict = {}

        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                img_path = full_path
                org_img = tools.img_cvread(img_path).copy()

                t1 = time.time()
                results = self.model.predict(org_img, conf=self.conf)
                detections = results[0].boxes

                car_count = 0
                person_count = 0
                boxes = []

                display_img = org_img.copy()

                if detections is not None:
                    for i, box in enumerate(detections):
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        score = box.conf[0].item()
                        cls = box.cls[0].item()
                        if score > self.conf:
                            # 获取英文类别名 → 转换为中文
                            en_cls = self.model.names[int(cls)]
                            ch_cls = Config2.EN_to_CH[en_cls]
                            # 更新计数并生成中文 display_name
                            class_counts[ch_cls] += 1
                            display_name = f"{ch_cls}{class_counts[ch_cls]}"
                            boxes.append((x1, y1, x2, y2, display_name, score))
                            # 绘制中文标签
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_img, f"{ch_cls}: {score:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(f"{take_time_str}")

                self.ui.total_lb.setText(f"{car_count + person_count}")
                self.ui.pedestrians_lb.setText(f"{person_count}")
                self.ui.vehicles_lb.setText(f"{car_count}")

                self.boxes_dict[img_path] = boxes

                self.ui.selection_cb.clear()
                self.ui.selection_cb.addItem("")
                for box in boxes:
                    self.ui.selection_cb.addItem(box[4], box)

                self.img_width, self.img_height = self.get_resize_size(display_img)
                resize_cvimg = cv2.resize(display_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)

                self.tabel_info_show(img_path)
                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()

                del org_img
                del display_img
                del results
                del detections
                del boxes

    def tabel_info_show(self, path):
        row_count = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(row_count)
        item_id = QTableWidgetItem(str(row_count + 1))
        item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        item_path = QTableWidgetItem(str(path))

        self.ui.tableWidget.setItem(row_count, 0, item_id)
        self.ui.tableWidget.setItem(row_count, 1, item_path)
        self.ui.tableWidget.scrollToBottom()

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Video files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        return file_path

    def video_start(self):
        self.video_index = 0
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, path):
        row_count = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(row_count)
        item_id = QTableWidgetItem(str(row_count + 1))
        item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        item_path = QTableWidgetItem(str(path))

        self.ui.tableWidget.setItem(row_count, 0, item_id)
        self.ui.tableWidget.setItem(row_count, 1, item_path)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        if self.cap:
            self.cap.release()
        self.timer_camera.stop()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            self.video_index += 1
            if self.video_index % self.detection_frequency == 0:
                t1 = time.time()
                try:
                    results = self.model(now_img, conf=self.conf)
                    boxes = results[0].boxes
                    target_count = 0  # 用于计算目标总数
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            score = box.conf[0].item()
                            cls = int(box.cls[0].item())
                            if cls >= len(self.model.names):  # 过滤非法ID
                                continue
                            label = f"{self.model.names[cls]} {score:.2f}"  # 使用 model.names
                            color = (0, 255, 0)  # 绿色

                            # 绘制检测框和显示标签
                            cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(now_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                        2)

                            target_count += 1  # 增加目标总数

                    # 更新UI显示检测结果数量
                    self.ui.total_lb.setText(f"{target_count}")

                    t2 = time.time()
                    take_time_str = '{:.3f} s'.format(t2 - t1)
                    self.ui.time_lb.setText(take_time_str)

                except Exception as e:
                    logging.error(f"Error during frame processing: {e}")

            # 显示当前帧
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 保存当前帧
            self.previous_frame = now_img
        else:
            logging.debug("No frame retrieved from the camera.")
            if hasattr(self, 'previous_frame'):
                now_img = self.previous_frame
                self.img_width, self.img_height = self.get_resize_size(now_img)
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)

        if not ret and self.cap:
            logging.debug("Releasing the video capture and stopping the timer.")
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CapBtn.setText('打开摄像头')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CapBtn.setText('关闭摄像头')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
        else:
            self.ui.CapBtn.setText('打开摄像头')
            self.ui.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width, depth = _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def on_cell_clicked(self, row, column):
        if self.cap:
            return

        img_path = self.ui.tableWidget.item(row, 1).text()
        self.org_img = tools.img_cvread(img_path).copy()  # 更新org_img
        display_img = self.org_img.copy()

        self.ui.type_lb.setText("")
        self.ui.confidence_lb.setText("")
        self.ui.xmin_lb.setText("")
        self.ui.ymin_lb.setText("")
        self.ui.xmax_lb.setText("")
        self.ui.ymax_lb.setText("")

        if img_path in self.boxes_dict:
            self.boxes = self.boxes_dict[img_path]
            car_count = 0
            person_count = 0

            for box in self.boxes:
                x1, y1, x2, y2, label, score = box
                if 'car' in label:
                    car_count += 1
                elif 'person' in label:
                    person_count += 1
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

            self.ui.total_lb.setText(f"{car_count + person_count}")
            self.ui.pedestrians_lb.setText(f"{person_count}")
            self.ui.vehicles_lb.setText(f"{car_count}")

        self.img_width, self.img_height = self.get_resize_size(display_img)
        resize_cvimg = cv2.resize(display_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

        self.update_selection_combobox()

    def update_selection_combobox(self):
        self.ui.selection_cb.clear()
        self.ui.selection_cb.addItem("")
        for box in self.boxes:
            self.ui.selection_cb.addItem(box[4], box)
        self.ui.selection_cb.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

