from PIL import Image
from tools import yolo_detect
import os
import sys

def test_yolo():
    # Test image path
    image_path = "/home/aditya/shashwath/YOLOv10/windowed_images/test1.png"
    
    # Load and process image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    # Run detection
    print("Running YOLO detection...")
    annotated_img, boxes = yolo_detect(image)
    
    # Save result
    print("Saving result...")
    output_path = "yolo_test_output.png"
    annotated_img.annotated_image.save(output_path)
    
    print(f"Number of detected boxes: {len(boxes)}")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

if __name__ == "__main__":
    test_yolo()