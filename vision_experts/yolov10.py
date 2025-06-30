import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLOv10 as YOLO
import torch
from PIL import Image

# Initialize YOLO model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('/home/aditya/shashwath/YOLOv10/yolov10/runs/detect/train/weights/best.pt')
model.to(DEVICE)

def predict_objects(input_image):
    """
    Process image through YOLOv10 model and draw bounding boxes
    
    Args:
        input_image: Input image (numpy array from Gradio)
    Returns:
        Image with drawn bounding boxes
    """
    # Convert to RGB if needed (Gradio might provide BGR)
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    else:
        image = input_image

    # Make a copy for drawing
    output_image = image.copy()
    
    try:
        # Perform inference using the same approach as your working code
        results = model(image)
        
        # Extract detection results
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()
        
        print(f"Number of detections: {len(boxes)}")  # Debug information
        
        # Draw bounding boxes and labels
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = names[int(cls)]
            confidence = conf
            
            # Print detection details (debug)
            print(f"Detection: {class_name}, Confidence: {confidence:.2f}")
            print(f"Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Draw rectangle (in RGB format)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{class_name}: {confidence:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(output_image, 
                         (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1),
                         (0, 255, 0), 
                         cv2.FILLED)
            
            # Draw label text
            cv2.putText(output_image,
                       label,
                       (x1, y1 - 5),
                       font,
                       font_scale,
                       (0, 0, 0),
                       thickness)
        
        if len(boxes) == 0:
            print("No detections found in the image")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return input_image
        
    # Convert back to BGR for OpenCV display if needed
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    return output_image

# Create Gradio interface with modified inputs
demo = gr.Interface(
    fn=predict_objects,
    inputs=[
        gr.Image(type="numpy", label="Input Image")  # Specify numpy type
    ],
    outputs=[
        gr.Image(type="numpy", label="Detected Objects")  # Specify numpy type
    ],
    title="YOLOv10 Object Detection",
    description="Upload an image to detect objects using YOLOv10. The model will detect and draw bounding boxes around detected objects.",
    examples=[
        # Add some example images if you have them
        # ["/path/to/example/image1.jpg"],
        # ["/path/to/example/image2.jpg"]
    ]
)

# Launch the server
if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=8082,
        share=True,
        debug=True  # Enable debug mode to see more information
    )