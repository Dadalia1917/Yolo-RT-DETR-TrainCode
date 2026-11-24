from ultralytics.nn.tasks import YOLOv10DetectionModel

from ..yolo import YOLO
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer
from .val import YOLOv10DetectionValidator


class YOLOv10(YOLO):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }
