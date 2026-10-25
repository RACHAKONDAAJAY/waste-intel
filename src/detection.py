
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


# Find the project folder and model file.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolov8n.pt"


def load_model():
    """Load the pretrained YOLOv8 model."""
    return YOLO(str(MODEL_PATH))


def detect_objects(image, confidence=0.10):
    """
    Detect common objects in an image.

    Args:
        image: A PIL Image.
        confidence: Detection confidence threshold from 0 to 1.

    Returns:
        annotated_image: PIL Image with detection boxes.
        detections: List of detected objects and confidence scores.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Please provide a PIL Image.")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    # Load the YOLOv8 model.
    model = load_model()

    # Convert image to RGB format.
    rgb_image = image.convert("RGB")
    image_array = np.array(rgb_image)

    # Run object detection.
    results = model.predict(
        source=image_array,
        conf=confidence,
        verbose=False,
    )

    result = results[0]
    detections = []

    # Collect detected objects.
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            class_name = result.names[class_id]

            coordinates = box.xyxy[0].tolist()

            detections.append(
                {
                    "object": class_name,
                    "confidence": round(score * 100, 2),
                    "x1": round(coordinates[0], 1),
                    "y1": round(coordinates[1], 1),
                    "x2": round(coordinates[2], 1),
                    "y2": round(coordinates[3], 1),
                }
            )

    # Draw bounding boxes around detected objects.
    annotated_array = result.plot()
    annotated_array = annotated_array[:, :, ::-1].copy()
    annotated_image = Image.fromarray(annotated_array)

    return annotated_image, detections


if __name__ == "__main__":
    print("YOLOv8 detection module is ready.")
    print("Model path:", MODEL_PATH)

    if MODEL_PATH.exists():
        print("YOLOv8 model file exists.")
    else:
        print(
            "The model file is not in the project folder. "
            "YOLOv8 may download the pretrained weights when loaded."
        )
