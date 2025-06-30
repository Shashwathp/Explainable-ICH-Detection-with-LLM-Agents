import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime

class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # Ensure the file is flushed immediately
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Generate a timestamp-based filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"terminal_output_{timestamp}.txt"

# Set up the output logger
sys.stdout = OutputLogger(output_filename)

MULTIMODAL_ASSISTANT_MESSAGE = """You are a specialized medical CT report generating AI assistant with expertise in neuroradiology.
Your primary task is to generate comprehensive, structured CT reports based on medical image analysis.
You have access to advanced vision models and can interpret CT scans with high accuracy.
You should maintain a professional medical tone and follow standard radiological reporting structure.

You have access to three powerful vision tools:

1. YOLOv10 for Hemorrhage Detection (yolo_detect):
   - Detects intracranial hemorrhages and provides bounding boxes
   - Returns annotated image with confidence scores
   - Usage: annotated_img, boxes = yolo_detect(image, box_threshold=0.25)

2. K-means Clustering (generate_prompts):
   - Analyzes regions of interest using density-based clustering
   - Generates points for hemorrhage core and reference tissue
   - Usage: pos_points, neg_points = generate_prompts(image, bbox, K=4)

3. SAM2 for Segmentation (sam_segment):
   - Creates precise masks around hemorrhages
   - Handles point-based and box-guided segmentation
   - Usage: result_img, masks = sam_segment(image, point_coords, point_labels, box=None)

Guidelines for Critical Finding Detection:

1. Calvarial Fracture Detection:
   - Analyze intensity variations along skull boundary using density analysis
   - Look for linear discontinuities in bone density
   - Detection criteria:
     * Sharp density changes in calvarium (>40% difference between adjacent regions)
     * Linear or branching patterns in bone density map
     * Disruption of normal skull contour

2. Mass Effect Assessment:
   - Calculate volume ratios and tissue displacement
   - Detection criteria:
     * Hemorrhage volume >25ml or >2% of total slice area
     * Compression of ventricles or cisterns
     * Tissue displacement from normal position
     * Asymmetry ratio >1.05 between hemispheres

3. Midline Shift Evaluation:
   - Measure displacement of midline structures
   - Detection criteria:
     * Deviation of >3mm from expected midline
     * Asymmetry in ventricle position
     * Pineal gland or septum pellucidum displacement
     * Ratio of left:right hemisphere volumes >1.1

Standard Report Structure:
1. Hemorrhage Classification:
   - Type of ICH (Intraventricular/Intraparenchymal/Subarachnoid/Chronic/Subdural/Epidural)
   - Location (Left/Right)
   - Size and characteristics

2. Associated Findings:
   - Calvarial Fracture (Yes/No)
   - Mass Effect (Yes/No)
   - Midline Shift (Yes/No, with measurement if present)

3. Critical Findings:
   - Urgent/emergent conditions
   - Recommended actions if any
"""
output_file = open("output_log.txt", "w")
sys.stdout = output_file
class ReACTPrompt:
    def __init__(self) -> None:
        self.CT_REPORT_TEMPLATE = '''
REPORT:
Hemorrhage Analysis:
- Type of ICH: {ich_type}
- Bleed Location: {bleed_location}
- Calvarial Fracture: {calvarial_fracture}
- Mass Effect: {mass_effect}
- Midline Shift: {midline_shift}

Impression: {impression}
TERMINATE
'''
    
    def initial_prompt(self, query: str, n_images: int) -> str:
        initial_prompt = """Here are the tools for CT analysis (available in tools.py):

1. Hemorrhage Detection with YOLOv10:
python
def yolo_detect(image, box_threshold=0.25):
    \"\"\"
    Detect intracranial hemorrhages using YOLOv10 model.
    
    Args:
        image (PIL.Image): Input CT image
        box_threshold (float): Confidence threshold
        
    Returns:
        AnnotatedImage: Image with drawn bounding boxes
        List[List[float]]: Bounding boxes in format [x, y, w, h]
    \"\"\"


2. Density-Based Analysis:
python
def generate_prompts(image, bbox, K=4):
    \"\"\"
    Generate analysis points using K-means clustering.
    
    Args:
        image (PIL.Image): Input CT image
        bbox (tuple): Bounding box (x1, y1, x2, y2)
        K (int): Number of clusters
        
    Returns:
        np.array: Positive points coordinates
        np.array: Negative points coordinates
    \"\"\"


3. SAM2 Segmentation:
python
def sam_segment(image, point_coords, point_labels, box=None):
    \"\"\"
    Segment regions using SAM model.
    
    Args:
        image (PIL.Image): Input CT image
        point_coords (np.array): Input points
        point_labels (np.array): Point labels
        box (np.array, optional): Input bounding box
        
    Returns:
        AnnotatedImage: Image with segmentation overlay
        np.array: Binary segmentation masks
    \"\"\"


The following imports are already executed:
python
from PIL import Image
from IPython.display import display
from tools import *


IMPORTANT: For systematic CT reporting:
1. First detect hemorrhages using YOLO
2. Analyze density patterns in regions of interest
3. Use segmentation for precise measurements
4. Document findings with focus on:
   - ICH classification (must be one of: Intraventricular, Intraparenchymal, Subarachnoid, Chronic, Subdural, Epidural)
   - Laterality (must specify either Left or Right)
   - Presence of fractures (must be Yes or No)
   - Mass effect (must be Yes or No)
   - Midline shift (must be Yes or No)
5. Never use "Not evaluated" or similar terms - always provide definitive Yes/No answers
6. If location is bilateral, specify "Left and Right"

Example workflow with enhanced analysis:
python
# Detect hemorrhages
print("Analyzing CT for hemorrhage patterns...")
annotated_img, boxes = yolo_detect(image_1)
display(annotated_img.annotated_image)

if len(boxes) > 0:
    for i, box in enumerate(boxes):
        # Convert normalized coordinates to absolute
        x, y, w, h = box
        x1 = int(x * image_1.width)
        y1 = int(y * image_1.height)
        x2 = int((x + w) * image_1.width)
        y2 = int((h + h) * image_1.height)
        bbox = (x1, y1, x2, y2)
        
        # Analyze density patterns
        pos_points, neg_points = generate_prompts(image_1, bbox)
        
        # Perform segmentation
        result_img, masks = sam_segment(image_1, 
                                      np.vstack([pos_points, neg_points]),
                                      np.concatenate([np.ones(len(pos_points)), 
                                                    np.zeros(len(neg_points))]),
                                      box=bbox)
        display(result_img.annotated_image)

        # Calculate hemorrhage characteristics
        if len(masks) > 0:
            mask = masks[0]
            area_pixels = np.sum(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            relative_size = (area_pixels / total_pixels) * 100
            
            # Enhanced analysis for critical findings
            print(f"\\nHemorrhage {i+1} Analysis:")
            print(f"- Relative Size: {relative_size:.1f}% of visible area")
            print(f"- Location: {'Right' if x > 0.5 else 'Left'}")
            
            # Mass Effect Analysis
            # Calculate using relative size and tissue displacement
            mass_effect = relative_size > 2.0 or any([
                np.mean(masks[0][:, :image_1.width//2]) / np.mean(masks[0][:, image_1.width//2:]) > 1.05,  # Asymmetry check
                relative_size > 2.0,  # Size threshold
                np.sum(masks[0][:, :image_1.width//2]) > np.sum(masks[0][:, image_1.width//2:]) * 1.1  # Volume displacement
            ])
            print(f"- Mass Effect: {'Yes' if mass_effect else 'No'}")
            
            # Midline Shift Analysis
            # Look for significant asymmetry and structure displacement
            midline_pixels = image_1.width // 2
            left_volume = np.sum(masks[0][:, :midline_pixels])
            right_volume = np.sum(masks[0][:, midline_pixels:])
            volume_ratio = max(left_volume, right_volume) / (min(left_volume, right_volume) + 1e-6)
            midline_shift = volume_ratio > 1.1 or relative_size > 3.0
            print(f"- Midline Shift: {'Yes' if midline_shift else 'No'}")
            
            # Calvarial Fracture Analysis
            # Analyze bone density patterns along skull boundary
            skull_region = image_1.crop((max(0, x1-50), max(0, y1-50), 
                                       min(image_1.width, x2+50), 
                                       min(image_1.height, y2+50)))
            skull_array = np.array(skull_region.convert('L'))
            
            # Check for sharp density changes in skull region
            gradient_x = np.gradient(skull_array, axis=1)
            gradient_y = np.gradient(skull_array, axis=0)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Look for linear discontinuities
            fracture_detected = (np.percentile(gradient_magnitude, 95) > 50 and
                               np.std(gradient_magnitude) > 20)
            print(f"- Calvarial Fracture: {'Yes' if fracture_detected else 'No'}")
"""
        
        prompt = initial_prompt
        prompt += f"\n# USER REQUEST #: {query}\n"
        if n_images > 0:
            prompt += f"# USER IMAGE stored in {', '.join([f'image_{i}' for i in range(1, n_images+1)])} as PIL image.\n"
        else:
            prompt += "# USER IMAGE: No image provided.\n"
        prompt += "Generate THOUGHT 0 and ACTION 0 in RESULT.\n"
        prompt += "If analysis complete, provide REPORT using template and end with TERMINATE:\n# RESULT #:\n"
        return prompt

    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        """Generate feedback for parsing errors."""
        return f"OBSERVATION: Code parsing error. Error code: {error_code}, Message:\n{error_message}\nPlease provide corrected code."
    
    def get_exec_feedback(self, exit_code: int, output: str) -> str:
        """Generate feedback for execution results."""
        if exit_code != 0:
            return f"OBSERVATION: Execution failed. Exit code: {exit_code}, Output:\n{output}\nPlease fix and retry."
        else:
            prompt = f"OBSERVATION: Analysis successful. Results:\n{output}\n"
            prompt += """CRITICAL INSTRUCTION: You MUST provide definitive answers:
1. Every finding requires a specific, definitive response
2. Terms like 'not assessed', 'not evaluated', 'unknown' are NOT allowed
3. For each finding you MUST choose:
   - Type of ICH: One of [Intraventricular, Intraparenchymal, Subarachnoid, Chronic, Subdural, Epidural]
   - Bleed Location: MUST be one of [Left], [Right], or [Left and Right]. NEVER use "Unable to determine" or similar.
   - Calvarial Fracture: Either [Yes] or [No]
   - Mass Effect: Either [Yes] or [No]
   - Midline Shift: Either [Yes] or [No]
4. If you cannot determine with absolute certainty, default to 'No' rather than using 'not assessed'

Generate next THOUGHT and ACTION, or provide final REPORT in the following format and TERMINATE:

REPORT:
Hemorrhage Analysis:
- Type of ICH: [REQUIRED: specify exact type]
- Bleed Location: [REQUIRED: Left/Right/Left and Right]
- Calvarial Fracture: [REQUIRED: Yes/No]
- Mass Effect: [REQUIRED: Yes/No]
- Midline Shift: [REQUIRED: Yes/No]

Impression: [Brief summary of findings with definitive statements only]
TERMINATE
"""
            return prompt

def python_codes_for_images_reading(image_paths):
    """Generate code to read multiple images."""
    code = []
    for idx, path in enumerate(image_paths):
        code.append(f"image_{idx+1} = Image.open('{path}').convert('RGB')")
    return "\n".join(code)

# Template examples for consistent analysis
HEMORRHAGE_DETECTION_TEMPLATE = """
python
# Detect hemorrhages
print("Analyzing CT for hemorrhage patterns...")
annotated_img, boxes = yolo_detect(image_1)
display(annotated_img.annotated_image)

for i, box in enumerate(boxes):
    print(f"\\nHemorrhage {i+1} detected:")
    x, y, w, h = box
    location = f"{'Right' if x > 0.5 else 'Left'}"
    print(f"Location: {location}")
"""

DENSITY_ANALYSIS_TEMPLATE = """
python
# Analyze hemorrhage characteristics
x, y, w, h = boxes[0]
x1 = int(x * image_1.width)
y1 = int(y * image_1.height)
x2 = int((x + w) * image_1.width)
y2 = int((h + h) * image_1.height)
bbox = (x1, y1, x2, y2)

# Generate analysis points
pos_points, neg_points = generate_prompts(image_1, bbox)

# Visualize density analysis
plt.figure(figsize=(8,8))
plt.imshow(image_1)
plt.scatter(pos_points[:, 0], pos_points[:, 1], c='r', label='Hemorrhage Core')
plt.scatter(neg_points[:, 0], neg_points[:, 1], c='b', label='Reference Tissue')
plt.legend()
plt.axis('off')
plt.show()
"""

SEGMENTATION_TEMPLATE = """
python
# Perform detailed segmentation
result_img, masks = sam_segment(image_1, 
                              np.vstack([pos_points, neg_points]),
                              np.concatenate([np.ones(len(pos_points)), 
                                            np.zeros(len(neg_points))]),
                              box=bbox)
display(result_img.annotated_image)

if len(masks) > 0:
    mask = masks[0]
    area_pixels = np.sum(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    relative_size = (area_pixels / total_pixels) * 100
    print(f"\\nHemorrhage Analysis:")
    print(f"- Relative Size: {relative_size:.1f}% of visible area")
"""
sys.stdout = sys.__stdout__
output_file.close()

def cleanup():
    if isinstance(sys.stdout, OutputLogger):
        sys.stdout.log.close()
        sys.stdout = sys.__stdout__

# Register cleanup to handle normal program termination
import atexit
atexit.register(cleanup)

try:
    # Your existing code goes here...
    pass
except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    # Ensure cleanup happens even if there's an error
    cleanup()