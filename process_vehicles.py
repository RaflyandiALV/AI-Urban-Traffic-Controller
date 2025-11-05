
import os
import re
from detect import run

# Vehicle classes for counting
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

def extract_vehicle_counts(yolo_output):
   
    vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}

    if isinstance(yolo_output, dict):
        # If the output is a dictionary, directly iterate over its items
        for cls, count in yolo_output.items():
            if cls in vehicle_counts:
                vehicle_counts[cls] += count
    elif isinstance(yolo_output, str):
        # If the output is a string, use regex to extract counts
        matches = re.findall(r"(\d+) (\w+)", yolo_output)
        for count, cls in matches:
            if cls in vehicle_counts:
                vehicle_counts[cls] += int(count)
    else:
        raise TypeError(f"Unsupported YOLOv5 output type: {type(yolo_output)}")

    return vehicle_counts

def detect_objects_for_all_remaining_roads(roads, scan_index):
   
    detected_counts = {}
    for road in roads:
        # Construct the folder path and image file based on the road and scan index
        road_number = road.split("_")[1]  # Extract road number (e.g., "1" from "road_1")
        folder_path = f"data/{road_number}.1"
        image_file = os.path.join(folder_path, f"image{scan_index}.jpg")

        # Check if the file exists before running detection
        if not os.path.exists(image_file):
            print(f"Warning: {image_file} does not exist. Skipping this road.")
            detected_counts[road] = {cls: 0 for cls in VEHICLE_CLASSES}
            continue

        print(f"Running object detection on file: {image_file}")
        yolo_output = run(
            weights="yolov5l.pt",          # Use the YOLOv5 large model
            source=image_file,            # Input image
            conf_thres=0.10,              # Confidence threshold
            iou_thres=0.2,                # IoU threshold for NMS
            device="0",                   # Run on GPU 0
            classes=[2, 3, 5, 7],         # Class indices for car, motorcycle, bus, and truck
            nosave=True                   # Do not save inference images
        )

        if yolo_output:
            # Extract vehicle counts from YOLOv5 output
            yolo_result = yolo_output[0] if isinstance(yolo_output, list) else yolo_output
            print(f"YOLOv5 Output: {yolo_result}")
            detected_counts[road] = extract_vehicle_counts(yolo_result)
        else:
            print(f"Warning: YOLOv5 returned no results for {image_file}")
            detected_counts[road] = {cls: 0 for cls in VEHICLE_CLASSES}

    return detected_counts