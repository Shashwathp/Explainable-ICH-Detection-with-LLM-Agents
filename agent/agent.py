from .mm_user_proxy_agent import MultimodalUserProxyAgent
from autogen.agentchat import Agent
from typing import Dict, Optional, Union
import os
import json
from datetime import datetime
import shutil

class SketchpadUserAgent(MultimodalUserProxyAgent):
    def __init__(
        self, 
        name,
        prompt_generator, 
        parser,
        executor,
        output_dir: str = "outputs",  # Directory to save outputs
        **config,
    ):
        super().__init__(name=name, **config)
        self.prompt_generator = prompt_generator
        self.parser = parser
        self.executor = executor
        self.output_dir = output_dir
        self.conversation_history = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _save_conversation(self, message: Union[Dict, str], sender: Agent):
        """Save each message in the conversation history."""
        timestamp = datetime.now().isoformat()
        message_data = {
            "timestamp": timestamp,
            "sender": sender.name,
            "message": message
        }
        self.conversation_history.append(message_data)
        
    def _save_execution_output(self, task_id: str, exit_code: int, output: str, file_paths: list):
        """Save execution outputs and files to the output directory."""
        # Create task-specific directory
        task_dir = os.path.join(self.output_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Save execution output
        output_data = {
            "exit_code": exit_code,
            "output": output,
            "files": file_paths,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(task_dir, "execution_output.json"), "w") as f:
            json.dump(output_data, f, indent=2)
            
        # Copy generated files to output directory
        if file_paths:
            files_dir = os.path.join(task_dir, "generated_files")
            os.makedirs(files_dir, exist_ok=True)
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, files_dir)

    def _save_task_completion(self, task_id: str):
        """Save final conversation and task summary."""
        task_dir = os.path.join(self.output_dir, f"task_{task_id}")
        
        # Save conversation history
        conversation_file = os.path.join(task_dir, "conversation_history.json")
        with open(conversation_file, "w") as f:
            json.dump(self.conversation_history, f, indent=2)
            
        # Save feedback types
        feedback_data = {
            "feedback_types": self.feedback_types,
            "timestamp": datetime.now().isoformat()
        }
        feedback_file = os.path.join(task_dir, "feedback_summary.json")
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)

    def sender_hits_max_reply(self, sender: Agent):
        return self._consecutive_auto_reply_counter[sender.name] >= self._max_consecutive_auto_reply

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Process received message."""
        print("COUNTER:", self._consecutive_auto_reply_counter[sender.name])
        self._process_received_message(message, sender, silent)
        
        # Save the received message
        self._save_conversation(message, sender)
        
        # parsing the code component
        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_status = parsed_results['status']
        parsed_error_message = parsed_results['message']
        parsed_error_code = parsed_results['error_code']
        
        # if TERMINATION message, then return
        if not parsed_status and self._is_termination_msg(message):
            self._save_task_completion(self.current_task_id)
            return
        
        # if parsing fails
        if not parsed_status:
            # reset the consecutive_auto_reply_counter
            if self.sender_hits_max_reply(sender):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
                
            self._consecutive_auto_reply_counter[sender.name] += 1
            reply = self.prompt_generator.get_parsing_feedback(parsed_error_message, parsed_error_code)
            self.feedback_types.append("parsing")
            self.send(reply, sender, request_reply=True)
            return
            
        # if parsing succeeds, execute the code
        if self.executor:
            exit_code, output, file_paths = self.executor.execute(parsed_content)
            
            # Save execution outputs
            self._save_execution_output(self.current_task_id, exit_code, output, file_paths)
            
            reply = self.prompt_generator.get_exec_feedback(exit_code, output)
            
            # if execution fails
            if exit_code != 0:
                if self.sender_hits_max_reply(sender):
                    self._consecutive_auto_reply_counter[sender.name] = 0
                    return
                    
                self._consecutive_auto_reply_counter[sender.name] += 1
                self.send(reply, sender, request_reply=True)
                return
                
            # if execution succeeds
            self.send(reply, sender, request_reply=True)
            self._consecutive_auto_reply_counter[sender.name] = 0
            return
    
    def generate_init_message(self, query, n_image):
        content = self.prompt_generator.initial_prompt(query, n_image)
        return content

    def initiate_chat(self, assistant, message, n_image, task_id, log_prompt_only=False):
        self.current_task_id = task_id
        self.feedback_types = []
        self.conversation_history = []  # Reset conversation history for new task
        
        initial_message = self.generate_init_message(message, n_image)
        if log_prompt_only:
            print(initial_message)
        else:
            # Save the initial message
            self._save_conversation(initial_message, self)
            assistant.receive(initial_message, self, request_reply=True)