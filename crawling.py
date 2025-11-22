import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)

model_full = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = nn.Sequential(
    model_full.features,               
    nn.AdaptiveAvgPool2d(1),     
    nn.Flatten()                 
)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_vector(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(img_t).squeeze().cpu().numpy()
    return vec

base_dir = r"D:\Crawling"

all_vectors = {}

for webtoon in tqdm(os.listdir(base_dir), desc="작품별 처리"):
    webtoon_path = os.path.join(base_dir, webtoon)
    if os.path.isdir(webtoon_path):  
        vectors = []
        for file in os.listdir(webtoon_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(webtoon_path, file)
                try:
                    vec = extract_vector(img_path)
                    print(f"[DEBUG] {img_path}: {vec[:5]}...")
                    vectors.append(vec)
                except Exception as e:
                    print("에러 발생:", img_path, e)

        if vectors:
            all_vectors[webtoon] = np.mean(vectors, axis=0)
            print(f"[DEBUG] {webtoon}: {len(vectors)}장 처리됨")
    all_vectors[webtoon] = np.array(vectors)

print("작품별 벡터 추출 완료")
print("저장된 작품 리스트:", all_vectors.keys())