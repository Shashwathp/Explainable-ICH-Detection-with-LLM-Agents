import os

# set up the agent
MAX_REPLY = 10

# set up the LLM for the agent
os.environ['OPENAI_API_KEY'] = ''  # Replace with your key
os.environ["AUTOGEN_USE_DOCKER"] = "False"
llm_config={
    "cache_seed": None, 
    "config_list": [
        {
            "model": "chatgpt-4o-latest", 
            "temperature": 0.0, 
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    ]
}

# Server addresses for the three vision tools
YOLO_SERVER = "http://localhost:8082/"  # YOLOv10 server
CLUSTERING_SERVER = "http://localhost:8083/"  # Clustering server
SAM2_SERVER = "http://localhost:8084/"  # SAM2 server