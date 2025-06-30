import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import datetime

def image_to_base64(pil_img):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte)
    return img_base64.decode('utf-8')

def custom_encoder(obj):
    """Custom JSON encoder that handles:
    - PIL Images (converts to base64)
    - NumPy arrays
    - Datetime objects
    - Other objects (converts to string)
    """
    if isinstance(obj, Image.Image):
        return image_to_base64(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    # Let the default JSON encoder handle other types
    try:
        return json.JSONEncoder().default(obj)
    except:
        return str(obj)