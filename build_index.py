import os
import torch
import faiss
import gc
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

IMAGE_FOLDER = "tshirt"
BATCH_SIZE = 8
EMB_FILE = "image_embeddings.npy"
PATHS_FILE = "image_paths.txt"
INDEX_FILE = "faiss.index"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

image_paths = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

all_embeddings = []

for i in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    images = []

    for p in batch_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))

    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        feats = model.get_image_features(**inputs)

    feats = feats / feats.norm(dim=1, keepdim=True)
    all_embeddings.append(feats.cpu().numpy())

    del images, inputs, feats
    torch.cuda.empty_cache()
    gc.collect()

embeddings = np.vstack(all_embeddings)

# Save everything
np.save(EMB_FILE, embeddings)
with open(PATHS_FILE, "w") as f:
    for p in image_paths:
        f.write(p + "\n")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

print("Indexing complete. Saved to disk.")
