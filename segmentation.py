import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
import cv2
import numpy as np
import time
import platform
import os

# Load the pretrained Mask R-CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# Preprocessing function
def preprocess(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0).to(device)

# Generate beep sound
def beep():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(440, 1500)  # 440 Hz for 1.5 seconds
    elif system == "Linux":
        os.system("beep -f 440 -l 1500")  # Beep at 440 Hz for 1.5 seconds
    else:
        print("\a")  # Fallback beep

# Mock GPS data
def get_gps_coordinates():
    # Replace this with actual GPS module integration
    return 12.971598, 77.594566  # Example: Latitude, Longitude of Bangalore, India

# Postprocessing: Thermal effect and bounding boxes
def overlay_thermal_effect(image, boxes, masks, labels):
    thermal_colormap = cv2.COLORMAP_JET  # Thermal-like colormap
    output_image = cv2.applyColorMap(image, thermal_colormap)
    overlay = output_image.copy()

    person_count = 0
    gps_lat, gps_lon = get_gps_coordinates()

    for i, (box, mask, label) in enumerate(zip(boxes, masks, labels)):
        if label == 1:  # Only process "person" labels
            person_count += 1
            color = (255, 255, 255)  # White bounding box for clarity
            binary_mask = mask > 0.5
            overlay[binary_mask] = (0.6 * overlay[binary_mask] + 0.4 * np.array(color)).astype(np.uint8)
            # Draw bounding box
            box = box.int().tolist()
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)
            # Add label
            cv2.putText(overlay, f"Person {i + 1}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display GPS coordinates and person count
    text = f"GPS: Lat {gps_lat:.6f}, Lon {gps_lon:.6f} | Persons: {person_count}"
    cv2.putText(
        overlay, text, (10, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # Beep sound if persons are detected
    if person_count > 0:
        beep()

    return overlay

# Main segmentation function
def segment_persons(frame):
    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess the frame
    input_image = preprocess(image)

    # Perform instance segmentation
    with torch.no_grad():
        predictions = model(input_image)[0]

    # Filter predictions for persons
    boxes = predictions['boxes']
    masks = predictions['masks']
    labels = predictions['labels']
    scores = predictions['scores']

    filtered_boxes, filtered_masks, filtered_labels = [], [], []
    for i in range(len(scores)):
        if scores[i] > 0.7 and labels[i] == 1:  # Confidence threshold and "person" label
            filtered_boxes.append(boxes[i])
            filtered_masks.append(masks[i][0])  # Single mask per person
            filtered_labels.append(labels[i])

    # Apply thermal effect and overlays
    result = overlay_thermal_effect(frame, filtered_boxes, filtered_masks, filtered_labels)
    return result