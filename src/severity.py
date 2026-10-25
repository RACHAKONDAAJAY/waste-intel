
"""
Waste Intel - Severity Priority Module

Combines image detection count and recorded waste quantity
to calculate a demonstration collection priority.
"""


def calculate_severity(detections, waste_kg=0.0):
    """
    Calculate severity using:
    1. Number of objects detected in an image
    2. Recorded waste quantity in kilograms

    Returns:
        Dictionary containing detection count, waste quantity,
        score, severity, and priority.
    """

    # Validate the waste quantity.
    try:
        waste_kg = float(waste_kg)
    except (TypeError, ValueError):
        raise ValueError("Waste quantity must be a valid number.")

    if waste_kg < 0:
        raise ValueError("Waste quantity cannot be negative.")

    # Count detected objects safely.
    if detections is None:
        object_count = 0
    elif isinstance(detections, (list, tuple)):
        object_count = len(detections)
    else:
        raise ValueError(
            "Detections must be a list, tuple, or None."
        )

    # Image detection score: 0 to 50.
    if object_count == 0:
        image_score = 0
    elif object_count <= 2:
        image_score = 15
    elif object_count <= 5:
        image_score = 30
    else:
        image_score = 50

    # Recorded waste score: 0 to 50.
    if waste_kg == 0:
        quantity_score = 0
    elif waste_kg <= 25:
        quantity_score = 10
    elif waste_kg <= 50:
        quantity_score = 20
    elif waste_kg <= 100:
        quantity_score = 35
    else:
        quantity_score = 50

    # Combined score ranges from 0 to 100.
    total_score = image_score + quantity_score

    # Assign a demonstration priority.
    if total_score == 0:
        severity = "No Priority"
        priority = 4
    elif total_score <= 30:
        severity = "Low"
        priority = 3
    elif total_score <= 60:
        severity = "Medium"
        priority = 2
    else:
        severity = "High"
        priority = 1

    return {
        "object_count": object_count,
        "waste_kg": waste_kg,
        "image_score": image_score,
        "quantity_score": quantity_score,
        "score": total_score,
        "severity": severity,
        "priority": priority,
    }
