# ml_service.py
from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load pretrained ResNet50
resnet = models.resnet50(pretrained=True)
model = nn.Sequential(*list(resnet.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)

@app.post("/get-embedding")
async def extract_embedding(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = get_embedding(contents).tolist()
    return {"embedding": embedding}

