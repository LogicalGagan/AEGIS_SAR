from abc import ABC, abstractmethod
import numpy as np

class Detector(ABC):
    @abstractmethod
    def infer(self, frame: np.ndarray) -> list[dict]:
        """
        Runs YOLOv8-nano inference on the provided frame.
        Returns a list of dictionaries containing bounding boxes and class labels.
        """
        pass
