"""Phone detection using TFLite SSD MobileNet V1 COCO.

No PyTorch/ultralytics required -- uses tflite-runtime (~5MB wheel)
with a 4MB quantized COCO model downloaded by setup.sh.
"""
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from . import config


class PhoneDetector:
    def __init__(self, model_path: str = config.TFLITE_MODEL):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self._inp = self.interpreter.get_input_details()[0]
        # Sort output tensors by index: boxes, classes, scores, count
        outs = sorted(self.interpreter.get_output_details(), key=lambda x: x["index"])
        self._out_boxes, self._out_cls, self._out_scores, self._out_count = outs

        self._inp_h = self._inp["shape"][1]
        self._inp_w = self._inp["shape"][2]

    def detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        """Returns list of (x1, y1, x2, y2, conf) for phones."""
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(
            cv2.resize(frame_bgr, (self._inp_w, self._inp_h)), cv2.COLOR_BGR2RGB
        )
        self.interpreter.set_tensor(self._inp["index"], np.expand_dims(rgb, 0))
        self.interpreter.invoke()

        # Outputs are dequantized floats even for uint8 model
        boxes   = self.interpreter.get_tensor(self._out_boxes["index"])[0]    # (N,4) ymin,xmin,ymax,xmax normalised
        classes = self.interpreter.get_tensor(self._out_cls["index"])[0]      # (N,)
        scores  = self.interpreter.get_tensor(self._out_scores["index"])[0]   # (N,)
        count   = int(self.interpreter.get_tensor(self._out_count["index"])[0])

        phones = []
        for i in range(count):
            if scores[i] < config.PHONE_CONF_THRESHOLD:
                continue
            if int(classes[i]) != config.PHONE_CLASS_ID:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            phones.append((
                int(xmin * w), int(ymin * h),
                int(xmax * w), int(ymax * h),
                float(scores[i]),
            ))
        return phones


def phone_near_face(face_bbox: tuple, phones: list[tuple], mult: float) -> bool:
    """True if any phone bbox is within `mult` * face_height of the face."""
    fx1, fy1, fx2, fy2 = face_bbox
    face_h = fy2 - fy1
    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2

    max_dist = mult * face_h

    for px1, py1, px2, py2, _ in phones:
        phone_cx = (px1 + px2) / 2
        phone_cy = (py1 + py2) / 2
        dist = ((phone_cx - face_cx) ** 2 + (phone_cy - face_cy) ** 2) ** 0.5
        if dist <= max_dist:
            return True
    return False
