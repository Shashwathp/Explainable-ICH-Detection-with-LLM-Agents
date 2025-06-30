import json
import os
import argparse, shutil
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .agent import SketchpadUserAgent
from .multimodal_conversable_agent import MultimodalConversableAgent
from .prompt import (
    ReACTPrompt,
    python_codes_for_images_reading,
    MULTIMODAL_ASSISTANT_MESSAGE
)
from .parse import Parser
from .execution import CodeExecutor
from .utils import custom_encoder
from .config import MAX_REPLY, llm_config

def filter_image_tags(message):
    """Remove image tags from the message."""
    if isinstance(message, str):
        # Remove <image> tags
        message = message.replace("<image>", "[Image output displayed]")
        # Remove img tags
        import re
        message = re.sub(r'<img[^>]*>', '[Image]', message)
    return message

def run_agent(task_input, output_dir, task_type="vision"):
    """Run the Visual Sketchpad agent on one task instance."""
    
    assert task_type == "vision"
    
    task_input = task_input.rstrip('/')
    task_directory = os.path.join(output_dir, os.path.basename(task_input))
    
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(task_input, task_directory, dirs_exist_ok=True)
    
    task_metadata = json.load(open(os.path.join(task_input, "request.json")))
    query = task_metadata['query']
    images = task_metadata['images']

    prompt_generator = ReACTPrompt()
    parser = Parser()
    executor = CodeExecutor(working_dir=task_directory, use_vision_tools=True)
    
    image_reading_codes = python_codes_for_images_reading(images)
    image_loading_result = executor.execute(image_reading_codes)
    if image_loading_result[0] != 0:
        raise Exception(f"Error loading images: {image_loading_result[1]}")
    
    user = SketchpadUserAgent(
        name="multimodal_user_agent",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg=lambda x: isinstance(x, str) and 'TERMINATE' in x,
        prompt_generator=prompt_generator,
        parser=parser,
        executor=executor
    )
    
    # Override receive method to filter image tags
    original_receive = user.receive
    def filtered_receive(message, *args, **kwargs):
        filtered_message = filter_image_tags(message)
        return original_receive(filtered_message, *args, **kwargs)
    user.receive = filtered_receive
    
    planner = MultimodalConversableAgent(
        name="planner",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=MAX_REPLY,
        is_termination_msg=lambda x: False,
        system_message=MULTIMODAL_ASSISTANT_MESSAGE,
        llm_config=llm_config
    )
    
    # Override receive method for planner
    original_planner_receive = planner.receive
    def filtered_planner_receive(message, *args, **kwargs):
        filtered_message = filter_image_tags(message)
        return original_planner_receive(filtered_message, *args, **kwargs)
    planner.receive = filtered_planner_receive
    
    try:
        user.initiate_chat(
            planner,
            n_image=len(images),
            task_id="testing_case",
            message=query,
            log_prompt_only=False,
        )
        all_messages = planner.chat_messages[user]
         
    except Exception as e:
        print(e)
        all_messages = {'error': str(e)}
    
    with open(os.path.join(task_directory, "output.json"), "w") as f:
        json.dump(all_messages, f, indent=4, default=custom_encoder)
        
    usage_summary = {
        'total': planner.client.total_usage_summary, 
        'actual': planner.client.actual_usage_summary
    }
    with open(os.path.join(task_directory, "usage_summary.json"), "w") as f:
        json.dump(usage_summary, f, indent=4)
        
    user.executor.cleanup()
    user.reset()
    planner.reset()