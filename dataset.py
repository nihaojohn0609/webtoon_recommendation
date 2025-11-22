import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np

# =================================================================
# === 코랩 환경 설정: 경로 및 학습 설정 (3차 학습용) ===
# =================================================================
# ⚠️ Google Drive 마운트 후 경로를 확인하여 수정해주세요.
NAVER_ROOT_DIR = r"D:\Crawling\Naver_Processed"
KAKAO_ROOT_DIR = r"D:\Crawling\Kakao_Processed"
PRETRAIN_PATH = r"D:\Webtoon_Models\webtoon_cnn_naver_finetuned_all.pt"
OUTPUT_MODEL_PATH = r"D:\Webtoon_Models\webtoon_cnn_naver_augmented_finetuned.pt" # 3차 학습 최종 모델 저장 경로

# ✅ Batch Size 최적화 테스트: 8 -> 16 -> 32 -> 64 순으로 변경하며 OOM 직전 값 사용
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5

# 데이터 분할 비율
VAL_RATIO = 0.2
TEST_RATIO = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 # 코랩 환경에 최적화

# =================================================================
# === 데이터셋 및 헬퍼 함수 === (이전 코드와 동일)
# =================================================================

# 학습용 전처리 (Augmentation 적용)
transform_train = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 검증/테스트용 전처리 (Augmentation 없이 기본 변환)
transform_basic = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def collect_image_paths(base_dirs):
    image_paths = []
    if not isinstance(base_dirs, list): base_dirs = [base_dirs]
    for base_dir in base_dirs:
        if not os.path.exists(base_dir): continue
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    return image_paths

def extract_label_and_platform(path):
    # 코랩 환경용 경로 구분자 '/'에 최적화
    normalized_path = path.replace('\\', '/')
    parts = normalized_path.split('/')
    work_id = parts[-3] if len(parts) >= 3 else "Unknown_Work"
    platform = "Unknown_Platform"
    if "Naver" in path or "naver" in path: platform = "Naver"
    elif "Kakao" in path or "kakao" in path: platform = "Kakao"
    return work_id, platform

class WebtoonFinetuneDataset(Dataset):
    def __init__(self, data_list, label_to_idx, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, platform = self.data_list[idx]

        try:
            img = Image.open(path).convert("RGB")
        except:
            return torch.zeros(3, 320, 320), -1, platform

        if self.transform:
            img = self.transform(img)

        work_id, _ = extract_label_and_platform(path)
        label = self.label_to_idx.get(work_id, -1)

        return img, label, platform


# =================================================================
# ✅ 메인 학습 블록
# =================================================================
if __name__ == '__main__':
    print(f"🔥 네이버 웹툰 작품 ID 분류 3차 미세 조정 시작 (Batch Size: {BATCH_SIZE}, Augmentation 적용)")
    print(f"✅ 사용 장치: {DEVICE}")

    # 1. 전체 이미지 경로 수집 및 3분할 (네이버 데이터만 사용)
    all_image_paths = collect_image_paths([NAVER_ROOT_DIR])
    random.shuffle(all_image_paths)

    total_size = len(all_image_paths)
    if total_size == 0:
        print("🚨 오류: 데이터 경로에서 이미지를 찾지 못했습니다. Google Drive 마운트 및 경로를 확인하세요.")
        exit()

    test_size = int(total_size * TEST_RATIO)
    val_size = int(total_size * VAL_RATIO)

    test_image_paths = all_image_paths[:test_size]
    val_image_paths = all_image_paths[test_size : test_size + val_size]
    train_image_paths = all_image_paths[test_size + val_size :]

    # 2. 라벨 정의 및 인덱스 매핑
    all_labels = set(extract_label_and_platform(p)[0] for p in all_image_paths)
    if "Unknown_Work" in all_labels: all_labels.remove("Unknown_Work")
    sorted_labels = sorted(list(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    num_classes = len(sorted_labels)
    print(f"작품 ID (클래스): {num_classes}개")
    print(f"데이터 크기: Train={len(train_image_paths)}, Val={len(val_image_paths)}, Test={len(test_image_paths)}")

    # 3. 데이터셋 및 데이터로더 준비
    def create_data_list(paths):
        return [(path, extract_label_and_platform(path)[1]) for path in paths]

    train_data_list = create_data_list(train_image_paths)
    val_data_list = create_data_list(val_image_paths)
    test_data_list = create_data_list(test_image_paths)

    train_dataset = WebtoonFinetuneDataset(train_data_list, label_to_idx, transform_train)
    val_dataset = WebtoonFinetuneDataset(val_data_list, label_to_idx, transform_basic)
    test_dataset = WebtoonFinetuneDataset(test_data_list, label_to_idx, transform_basic)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 4. 모델 로드 및 학습 준비
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    if PRETRAIN_PATH and os.path.exists(PRETRAIN_PATH):
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE))
        print(f"✅ 2차 학습된 모델 가중치 로드 완료: {PRETRAIN_PATH}")

    for param in model.parameters(): param.requires_grad = True
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_acc = 0.0

    # 5. 학습 루프
    print(f"🔥 학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        # tqdm 대신 print로 진행 상황 출력 (코랩 환경에서 tqdm 대신 사용할 수 있음)
        for batch_idx, (imgs, labels, _) in enumerate(train_loader):
            valid_mask = (labels != -1)
            imgs = imgs[valid_mask].to(DEVICE)
            labels = labels[valid_mask].to(DEVICE)

            if len(labels) == 0: continue

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if (batch_idx + 1) % 100 == 0: # 100 배치마다 진행 상황 출력
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_acc = correct / total if total > 0 else 0

        # === 검증 (Validation) ===
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                valid_mask = (labels != -1)
                imgs = imgs[valid_mask].to(DEVICE)
                labels = labels[valid_mask].to(DEVICE)

                if len(labels) == 0: continue

                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"--- Epoch [{epoch+1}/{EPOCHS}] 완료 --- | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print(f"⭐ Best Model Saved! Val Acc: {best_val_acc:.4f}")

    # 6. 최종 테스트 (최종 모델 사용)
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location=DEVICE))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            valid_mask = (labels != -1)
            imgs = imgs[valid_mask].to(DEVICE)
            labels = labels[valid_mask].to(DEVICE)

            if len(labels) == 0: continue

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += len(labels)

    final_test_acc = test_correct / test_total if test_total > 0 else 0
    print(f"\n--- 3차 학습 최종 결과 ---")
    print(f"✨ 최종 Test 정확도: {final_test_acc:.4f}")
    print(f"모델 저장 위치: {OUTPUT_MODEL_PATH}")