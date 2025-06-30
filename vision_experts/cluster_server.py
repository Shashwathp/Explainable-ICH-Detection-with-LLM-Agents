import gradio as gr
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def generate_prompts(image, bbox, K=4):
    """Generate positive and negative points using K-means clustering."""
    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = image[y1:y2, x1:x2]
    
    # Flatten the image for clustering
    flat_image = cropped_image.reshape(-1, 3)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=K, random_state=42).fit(flat_image)
    clustered_image = kmeans.labels_.reshape(cropped_image.shape[:2])
    
    # Find brightest cluster
    cluster_means = [np.mean(flat_image[kmeans.labels_ == i]) for i in range(K)]
    sorted_clusters = np.argsort(cluster_means)[::-1]
    
    lesion_cluster = sorted_clusters[0]
    other_clusters = sorted_clusters[1:]
    
    # Generate points
    lesion_points = np.argwhere(clustered_image == lesion_cluster)
    positive_point = lesion_points[len(lesion_points) // 2]
    
    negative_points = []
    for cluster in other_clusters:
        points = np.argwhere(clustered_image == cluster)
        if points.size > 0:
            negative_points.append(points[len(points) // 2])
    
    # Convert to absolute coordinates
    positive_point_abs = (positive_point[0] + y1, positive_point[1] + x1)
    negative_points_abs = [(pt[0] + y1, pt[1] + x1) for pt in negative_points]
    
    return np.array([positive_point_abs]), np.array(negative_points_abs)

def process_image(input_image, x1, y1, x2, y2, num_clusters=4):
    """Process image and visualize clustering results."""
    try:
        # Convert to RGB if needed
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        else:
            image = input_image
            
        # Create bbox tuple
        bbox = (float(x1), float(y1), float(x2), float(y2))
        
        # Generate points
        positive_points, negative_points = generate_prompts(image, bbox, K=num_clusters)
        
        # Create visualization
        output_image = image.copy()
        
        # Draw bounding box
        cv2.rectangle(output_image, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     (0, 255, 0), 
                     2)
        
        # Draw positive points (red)
        for point in positive_points:
            cv2.circle(output_image, 
                      (int(point[1]), int(point[0])), 
                      5, 
                      (255, 0, 0), 
                      -1)
        
        # Draw negative points (blue)
        for point in negative_points:
            cv2.circle(output_image, 
                      (int(point[1]), int(point[0])), 
                      5, 
                      (0, 0, 255), 
                      -1)
        
        # Print points for reference
        print("Positive Points:", positive_points)
        print("Negative Points:", negative_points)
        
        return output_image
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return input_image

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Number(label="x1", value=105.96402740478516),
        gr.Number(label="y1", value=162.6252899169922),
        gr.Number(label="x2", value=173.80462646484375),
        gr.Number(label="y2", value=223.6249542236328),
        gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Number of Clusters")
    ],
    outputs=[
        gr.Image(type="numpy", label="Processed Image")
    ],
    title="K-means Clustering Analysis",
    description="Upload an image and provide bounding box coordinates to perform clustering analysis.",
    examples=[
        # ["/path/to/example/image.png", 105.96402740478516, 162.6252899169922, 173.80462646484375, 223.6249542236328, 4]
    ]
)

if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=8083,
        share=True,
        debug=True
    )