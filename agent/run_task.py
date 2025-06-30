import os
import glob
import argparse
from tqdm import tqdm
import sys

# Add the parent directory to sys.path to allow imports from the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the package using absolute imports
from agent.main import run_agent

def run_task(task, output_dir, task_type="vision", task_name=None):
    """Run a specific task on all its instances."""
    # Use absolute path for tasks directory
    tasks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")
    all_task_instances = glob.glob(os.path.join(tasks_dir, task, "processed", "*", ""))
    output_dir = os.path.join(output_dir, task)
    
    for task_instance in tqdm(all_task_instances):
        print(f"Running task instance: {task_instance}")
        run_agent(task_instance, output_dir, task_type=task_type, task_name=task_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", 
        type=str, 
        choices=[
            "medical_segmentation",
            "lesion_detection", 
            "tumor_analysis"
        ],
        help="The task name"
    )
    args = parser.parse_args()
    
    # All tasks use vision tools
    task_type = "vision"
    task_name = None
    
    run_task(args.task, "outputs", task_type=task_type, task_name=task_name)