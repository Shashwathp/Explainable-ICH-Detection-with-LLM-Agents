import os
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path('/home/aditya/shashwath/Med_VisualSketchpad1')

from PIL import Image
import numpy as np
import tempfile
import cv2
from ultralytics import YOLO, SAM
from sklearn.cluster import KMeans
import torch

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO('/home/aditya/shashwath/YOLOv10/yolov10/runs/detect/train/weights/best.pt')
yolo_model.to(device)
# Configure model parameters
yolo_model.conf = 0.25  # Set confidence threshold
yolo_model.mode = 'predict'
yolo_model.task = 'detect'
# SAM model
sam_model = SAM('/home/aditya/shashwath/SAM/sam2.1_b.pt')
sam_model.to(device)

class AnnotatedImage:
    def __init__(self, annotated_image: Image.Image, original_image: Image.Image=None):
        self.annotated_image = annotated_image
        self.original_image = original_image

# In tools.py, update the YOLO configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO model with explicit configuration
yolo_model = YOLO('/home/aditya/shashwath/YOLOv10/yolov10/runs/detect/train/weights/best.pt')
yolo_model.to(device)

def yolo_detect(image, box_threshold=0.25):
    """Detect objects using YOLOv10."""
    # Debug print
    print(f"Input image size: {image.size}")
    print(f"Input image mode: {image.mode}")
    
    # Save image to temporary file to match command-line behavior
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        image.save(tmp_file.name)
        print(f"Saved temporary image to: {tmp_file.name}")
        
        try:
            # Run inference using the same parameters as command line
            results = yolo_model.predict(
                source=tmp_file.name,
                conf=box_threshold,
                save=False,
                verbose=True,
                stream=False
            )
            
            # Process results
            output_image = np.array(image).copy()
            boxes = []
            
            if len(results) > 0:
                result = results[0]  # Get first result
                print(f"Number of detections: {len(result.boxes)}")
                print(f"Confidence scores: {result.boxes.conf.tolist()}")
                
                for box, cls, conf in zip(result.boxes.xyxy.tolist(), 
                                        result.boxes.cls.tolist(),
                                        result.boxes.conf.tolist()):
                    if conf > box_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cls_name = result.names[int(cls)]
                        cv2.putText(output_image, f"{cls_name}: {conf:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        w, h = x2-x1, y2-y1
                        boxes.append([x1/output_image.shape[1], y1/output_image.shape[0], 
                                   w/output_image.shape[1], h/output_image.shape[0]])
            
            return AnnotatedImage(Image.fromarray(output_image), image), boxes
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file.name)

def generate_prompts(image, bbox, K=4):
    """Generate positive and negative points using K-means clustering."""
    image_np = np.array(image)
    
    # Convert relative bbox coordinates to absolute
    w, h = image_np.shape[1], image_np.shape[0]
    x1, y1, x2, y2 = map(int, [
        max(0, bbox[0]), 
        max(0, bbox[1]), 
        min(w, bbox[2]), 
        min(h, bbox[3])
    ])
    
    # Crop the image
    cropped = image_np[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError(f"Invalid bounding box: {bbox} for image size {(h, w)}")
    
    # Convert to grayscale for clustering
    if len(cropped.shape) == 3:
        cropped_gray = np.mean(cropped, axis=2)
    else:
        cropped_gray = cropped
        
    # Flatten and normalize
    flat_image = cropped_gray.reshape(-1, 1) / 255.0
    
    # Perform K-means with fixed number of initializations
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(flat_image)
    
    # Reshape labels back to image shape
    clustered = kmeans.labels_.reshape(cropped_gray.shape)
    
    # Find brightest and darkest clusters
    cluster_means = [np.mean(flat_image[kmeans.labels_ == i]) for i in range(K)]
    sorted_clusters = np.argsort(cluster_means)
    
    # Get one positive point from brightest cluster and three negative points from others
    pos_cluster = sorted_clusters[-1]
    neg_clusters = sorted_clusters[:-1]
    
    # Get center point from positive cluster
    pos_points = np.argwhere(clustered == pos_cluster)
    if len(pos_points) > 0:
        pos_point = pos_points[len(pos_points) // 2]
    else:
        pos_point = np.array([(y2-y1)//2, (x2-x1)//2])
        
    # Get three evenly spaced points from negative clusters
    neg_points = []
    for cluster in neg_clusters[:3]:  # Take up to 3 negative clusters
        points = np.argwhere(clustered == cluster)
        if len(points) > 0:
            idx = len(points) // 2
            neg_points.append(points[idx])
            
    # Convert back to absolute coordinates
    pos_point_abs = np.array([[pos_point[0] + y1, pos_point[1] + x1]])
    neg_points_abs = np.array([[p[0] + y1, p[1] + x1] for p in neg_points])
    
    return pos_point_abs, neg_points_abs

def sam_segment(image, point_coords, point_labels, box=None):
    """Segment image using Ultralytics SAM."""
    
    # Save image to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        image.save(tmp_file.name)
        
        # Format points and labels
        points_list = [[[int(x), int(y)] for x, y in point_coords]]
        labels_list = [[int(label) for label in point_labels]]
        
        # Format box if provided
        bboxes = None
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            bboxes = [[x1, y1, x2, y2]]
        
        # Run inference
        results = sam_model(
            source=tmp_file.name,
            points=points_list,
            labels=labels_list,
            bboxes=bboxes,
            save=False  # Don't save to disk
        )
        
        # Get mask and create visualization
        masks = results[0].masks.data.cpu().numpy()  # Get binary masks
        
        # Create visualization
        output_image = np.array(image).copy()
        
        # Draw box if provided
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw points
        for point, label in zip(point_coords, point_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(output_image, 
                      (int(point[0]), int(point[1])), 
                      5, color, -1)
        
        # Add mask overlay
        if len(masks) > 0:
            mask = masks[0]  # Take first mask
            color_overlay = np.array([30, 144, 255])
            output_image = output_image.astype(float)
            output_image[mask] = output_image[mask] * 0.7 + color_overlay * 0.3
            output_image = output_image.astype(np.uint8)
        
        # Clean up temporary file
        os.unlink(tmp_file.name)
        
        return AnnotatedImage(Image.fromarray(output_image), image), masks