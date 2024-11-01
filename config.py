# Configurations (e.g., API keys, database URL)

## For storing uploaded file in to local folder.
app.config['UPLOAD_FOLDER'] = "uploads/"


## For Hugging Face APIs.
from dotenv import load_dotenv
import os

# Load the environment variables from the custom file
load_dotenv("config.env")

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_ENDPOINT_URL = os.getenv("HUGGINGFACE_ENDPOINT_URL")

print("HUGGINGFACE_API_KEY:", HUGGINGFACE_API_KEY)
print("HUGGINGFACE_ENDPOINT_URL:", HUGGINGFACE_ENDPOINT_URL)


