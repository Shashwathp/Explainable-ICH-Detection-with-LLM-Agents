import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path('/home/aditya/shashwath/Med_VisualSketchpad1')
sys.path.insert(0, str(PROJECT_ROOT))

from agent.main import run_agent

def process_all_cases():
    # Base input and output directories
    base_input_dir = PROJECT_ROOT / "tasks/medical_segmentation/processed"
    base_output_dir = PROJECT_ROOT / "outputs/medical_segmentation"
    
    # Find all val_case directories
    val_cases = [d for d in base_input_dir.iterdir() if d.is_dir() and d.name.startswith('val_case_')]
    
    print(f"Found {len(val_cases)} validation cases to process")
    
    # Process each case
    for case_dir in sorted(val_cases):  # sorted to process in order
        print(f"\nProcessing {case_dir.name}")
        print(f"Task input directory: {case_dir}")
        print(f"Output directory: {base_output_dir / case_dir.name}")
        
        # Run agent for each case
        run_agent(
            task_input=str(case_dir),
            output_dir=str(base_output_dir),
            task_type="vision"
        )

if __name__ == "__main__":
    process_all_cases()    