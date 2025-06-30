# Explainable ICH Detection with LLM Agents

## Overview

This repository contains the implementation of an explainable multimodal framework for intracranial hemorrhage (ICH) detection using Large Language Model (LLM) agents. The system combines YOLOv10, SAM2, and clustering techniques with GPT-4o cooperative agents to generate comprehensive diagnostic reports from CT scans.
![X_FLOW (2)](https://github.com/user-attachments/assets/c6e53dbc-5eaf-464b-9cfc-9c6aa313d9a2)


## Key Features

- **Multi-Agent Architecture**: Cooperative LLM agents (Multi-modal User Agent & Planner) for intelligent CT analysis
- **Advanced Vision Tools**: Integration of YOLOv10 for detection, SAM2 for segmentation, and K-means clustering
- **Explainable AI**: Chain-of-thought reasoning providing transparent diagnostic decisions
- **Clinical Parameters**: Automated assessment of bleed location, mass effect, midline shift, and calvarial fractures
- **Cost-Effective**: Strategic slice selection reducing inference cost to ~$0.70 per patient

## Performance

- **Detection Accuracy**: 94% on hemorrhage detection
- **Clinical Parameters**: 78.1% overall accuracy
- **Precision**: 78.0% | **Recall**: 74.4%

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU 
- 16GB+ RAM

### Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/explainable-ich-detection-with-llm-agents.git
cd explainable-ich-detection-with-llm-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install opencv-python
pip install scikit-learn
pip install autogen-agentchat
pip install openai
pip install gradio
pip install pillow
pip install numpy
pip install matplotlib
pip install omegaconf
pip install pyautogen
```

### Model Weights

1. **YOLOv10 Model**: Download the fine-tuned ICH detection weights
   ```bash
   # Place your YOLOv10 weights at:
   # /path/to/your/yolov10/weights/best.pt
   ```

2. **SAM2 Model**: Download SAM2 weights
   ```bash
   # Place SAM2 weights at:
   # /path/to/your/sam2/weights/sam2.1_b.pt
   ```

### Environment Configuration

1. Set up OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

2. Update model paths in `agent/config.py`:
   ```python
   # Update paths in config.py
   YOLO_MODEL_PATH = "/path/to/your/yolov10/weights/best.pt"
   SAM2_MODEL_PATH = "/path/to/your/sam2/weights/sam2.1_b.pt"
   ```

## Usage

### Quick Start

1. **Start Vision Expert Servers**:
   ```bash
   # Terminal 1: Start YOLOv10 server
   cd vision_experts
   python yolov10.py

   # Terminal 2: Start Clustering server
   python cluster_server.py

   # Terminal 3: Start SAM2 server (if separate)
   # Configure SAM2 server based on your setup
   ```

2. **Run ICH Analysis**:
   ```bash
   # Process a single case
   python agent/main.py --task_input /path/to/ct/images --output_dir outputs

   # Process multiple validation cases
   python agent/quick_start_vision.py
   ```

### Processing CT Scans

```python
from agent.main import run_agent

# Analyze CT scan
run_agent(
    task_input="path/to/ct/case",
    output_dir="outputs",
    task_type="vision"
)
```

### Input Format

Organize your CT data as follows:
```
task_input/
├── request.json          # Contains query and image paths
└── CT_slice.png         # CT scan slice(s)
```

Example `request.json`:
```json
{
    "query": "Detect and segment any medical abnormalities in this image using YOLOv10 for detection, clustering for point generation, and SAM for segmentation.",
    "images": ["CT_slice.png"]
}
```

## Architecture

### Multi-Agent Framework

1. **Multi-modal User Agent (MUA)**: 
   - Executes vision tools (YOLOv10, SAM2, clustering)
   - Processes CT images and extracts findings
   - Manages code execution and tool integration

2. **Planner Agent**:
   - Orchestrates analysis workflow
   - Makes decisions on tool usage
   - Generates diagnostic summaries

### Vision Tools Pipeline

1. **YOLOv10 Detection**: Identifies hemorrhage regions with confidence scores
2. **K-means Clustering**: Generates positive/negative points for segmentation
3. **SAM2 Segmentation**: Creates precise hemorrhage masks
4. **Clinical Analysis**: Evaluates mass effect, midline shift, fractures

## Datasets

### BHX Dataset
- 15,979 expert-annotated bounding boxes
- 6 ICH categories: Intraventricular, Intraparenchymal, Subarachnoid, Chronic, Subdural, Epidural
- Split: 80% train, 10% validation, 10% test

### Seg-CQ500 Dataset
- 51 scans with comprehensive clinical labels
- Includes midline shift, bleed location, fracture, mass effect annotations
- Used for pipeline validation

## Results

### Detection Performance

| Model | Precision | Recall | F1-Score | mAP@.5:.95 |
|-------|-----------|--------|----------|------------|
| **Ours** | **0.938** | **0.946** | **0.941** | **0.758** |
| YOLOv5s-CAM | 0.935 | 0.908 | 0.921 | 0.650 |
| GA (TL-LFF Net) | 0.935 | 0.945 | 0.940 | 0.728 |

### Clinical Parameters

| Parameter | Accuracy | Precision | Recall |
|-----------|----------|-----------|--------|
| Bleed Location | 0.765 | 0.720 | 0.683 |
| Calvarial Fracture | 0.798 | 0.675 | 0.742 |
| Mass Effect | 0.812 | 0.892 | 0.754 |
| Midline Shift | 0.749 | 0.831 | 0.796 |
| **Average** | **0.781** | **0.780** | **0.744** |

## Configuration

### Key Configuration Files

- `agent/config.py`: Model paths, API keys, server addresses
- `agent/prompt.py`: System prompts and templates
- `agent/tools.py`: Vision tool implementations

### Customization

1. **Model Weights**: Update paths in config files
2. **Prompts**: Modify system messages in `prompt.py`
3. **Clinical Parameters**: Adjust analysis criteria in tool functions
4. **Server Ports**: Configure in vision expert scripts

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce batch size or use CPU inference
2. **API Key Error**: Ensure OpenAI API key is properly set
3. **Model Loading**: Verify model weight paths are correct
4. **Server Connection**: Check if vision expert servers are running

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built on the AutoGen framework for multi-agent systems
- Utilizes YOLOv10 and SAM2 for computer vision tasks
- Inspired by Visual Sketchpad and ViperGPT methodologies

## Contact

For questions and support, please open an issue in this repository.
