"""COCO class catalog used by the live object toggles."""

COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

LABEL_OVERRIDES = {
    60: "Table",
    62: "Monitor",
}

GROUPS: list[tuple[str, list[int]]] = [
    ("People", [0]),
    ("Furniture", [13, 56, 57, 59, 60, 61]),
    ("Electronics", [62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 74]),
]

# Person, chair, table, laptop, monitor.
DEFAULT_ENABLED = (0, 56, 60, 62, 63)


def label_for(class_id: int) -> str:
    if class_id in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[class_id]
    return COCO_NAMES[class_id].title()


def catalog(enabled: set[int]) -> list[dict]:
    used = {class_id for _name, ids in GROUPS for class_id in ids}
    other = [class_id for class_id in COCO_NAMES if class_id not in used]
    sections = [*GROUPS, ("Other", other)]
    groups = []
    for name, ids in sections:
        groups.append(
            {
                "name": name,
                "items": [
                    {
                        "id": class_id,
                        "name": COCO_NAMES[class_id],
                        "label": label_for(class_id),
                        "enabled": class_id in enabled,
                    }
                    for class_id in ids
                ],
            }
        )
    return groups
