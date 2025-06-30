import os
import sys
from pathlib import Path

def test_tools():
    # Set up paths
    project_root = Path('/home/aditya/shashwath/Med_VisualSketchpad1')
    sys.path.insert(0, str(project_root))
    
    # Import tools
    from agent.tools import yolo_detect, generate_prompts, sam_segment  # Note: sam_segment instead of sam2_segment
    from PIL import Image
    import numpy as np
    
    # Test image
    image_path = "/home/aditya/shashwath/Conversion/rgbImages/file0001.png"
    image = Image.open(image_path)
    
    print("Testing YOLO detection...")
    annotated_img, boxes = yolo_detect(image)
    annotated_img.annotated_image.save('yolo_output.png')
    print(f"Detected boxes: {boxes}\n")
    
    print("Testing K-means clustering...")
    bbox = (105.96402740478516, 162.6252899169922, 173.80462646484375, 223.6249542236328)
    pos_points, neg_points = generate_prompts(image, bbox)
    print(f"Positive points: {pos_points}")
    print(f"Negative points: {neg_points}\n")
    
    print("Testing SAM segmentation...")
    input_point = np.array([[170, 117]])
    input_label = np.array([0])
    try:
        annotated_img, masks = sam_segment(image, input_point, input_label, box=bbox)
        annotated_img.annotated_image.save('sam_output.png')
        if masks is not None:
            print(f"Mask shape: {masks.shape}")
        else:
            print("No masks generated")
    except Exception as e:
        print(f"Error in SAM segmentation: {str(e)}")
    print()

if __name__ == "__main__":
    test_tools()