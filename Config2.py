# coding:utf-8

# 模型路径
model_path = 'runs/detect/train_v13/weights/best.pt'

# 英文类别名（需与训练时的 data.yaml 完全一致）
names = {
    0: 'harbor', 1: 'ship', 2: 'storagetank', 3: 'chimney', 4: 'dam',
    5: 'trainstation', 6: 'basketballcourt', 7: 'airport', 8: 'expressway-service-area',
    9: 'airplane', 10: 'baseballfield', 11: 'expressway-toll-station', 12: 'vehicle',
    13: 'golffield', 14: 'bridge', 15: 'groundtrackfield', 16: 'overpass',
    17: 'windmill', 18: 'tenniscourt', 19: 'stadium'
}

# 中文类别名（与 names 一一对应）
CH_names = [
    '港口', '船舶', '储油罐', '烟囱', '水坝', '火车站', '篮球场', '机场', '高速服务区',
    '飞机', '棒球场', '高速收费站', '车辆', '高尔夫场', '桥梁', '田径场', '立交桥', '风车',
    '网球场', '体育场'
]

# 提供双向映射（英文 → 中文，中文 → 英文）
EN_to_CH = {en: ch for en, ch in zip(names.values(), CH_names)}
CH_to_EN = {ch: en for en, ch in zip(names.values(), CH_names)}