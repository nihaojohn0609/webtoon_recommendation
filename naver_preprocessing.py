"""
웹툰 컷 이미지 전처리 파이프라인 (로컬 환경 - 완전판)
- 배경색 보존 적응형 리사이즈
- 과도한 여백 필터링
- pHash 기반 중복 제거 (선택)
- 한글 경로 완벽 지원
- 로컬 디렉토리 저장

필요 라이브러리 설치:
pip install pillow imagehash opencv-python tqdm numpy
"""

# ==================== 1. 환경 설정 ====================
import cv2
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import zipfile
import hashlib


# ==================== 2. 전처리 함수 정의 ====================

def estimate_background_color(image, border_width=5):
    """이미지 가장자리 픽셀의 중앙값으로 배경색 추정"""
    h, w = image.shape[:2]
    edges = np.concatenate([
        image[:border_width, :].reshape(-1, 3),
        image[-border_width:, :].reshape(-1, 3),
        image[:, :border_width].reshape(-1, 3),
        image[:, -border_width:].reshape(-1, 3)
    ])
    return np.median(edges, axis=0).astype(np.uint8)


def filter_excessive_spacing(image, max_spacing_ratio=0.7):
    """과도한 여백 컷 필터링 (크레딧/광고 제거)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 배경색 영역 비율 계산
    bg_color = estimate_background_color(image)
    bg_gray = np.mean(cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY))
    bg_mask = np.abs(gray - bg_gray) < 20
    spacing_ratio = np.mean(bg_mask)
    
    # 여백이 70% 이상이면 제외
    if spacing_ratio > max_spacing_ratio:
        return None, spacing_ratio
    
    return image, spacing_ratio


def smart_crop_vertical_spacing(image, min_content_height=50):
    """상하 과도한 여백만 제거 (좌우는 유지)"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 행별 콘텐츠 밀도 계산
    row_content = np.mean(gray, axis=1)
    bg_value = np.median([row_content[0], row_content[-1]])
    content_rows = np.abs(row_content - bg_value) > 10
    
    # 콘텐츠 있는 행 범위 찾기
    content_indices = np.where(content_rows)[0]
    if len(content_indices) == 0:
        return image
    
    top = max(0, content_indices[0] - 5)  # 여유 5px
    bottom = min(h, content_indices[-1] + 5)
    
    # 최소 높이 보장
    if bottom - top < min_content_height:
        return image
    
    return image[top:bottom, :]


def adaptive_resize_with_original_bg(image, target_size=320):
    """
    모든 이미지를 TARGET_SIZE x TARGET_SIZE 정방형으로 통일. (비율 보존)
    """
    h, w = image.shape[:2]
    
    # 1. 🎯 긴 변을 target_size에 맞추어 비율 유지
    if h > w:
        scale = target_size / h 
    else:
        scale = target_size / w 
        
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 리사이즈 시 0 크기 방지 및 리사이즈
    if new_h == 0 or new_w == 0:
        new_h = max(1, new_h); new_w = max(1, new_w) # 최소 1px 보장
        
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. 배경색 자동 추정
    bg_color = estimate_background_color(image)
    bg_color = tuple(map(int, bg_color)) 
    
    # 3. TARGET_SIZE x TARGET_SIZE 정사각형으로 패딩
    # 패딩 목표 크기는 target_size
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    padded = cv2.copyMakeBorder(
        resized, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, 
        value=bg_color
    )
    
    # 최종 결과는 항상 target_size x target_size
    return padded


def perceptual_hash_with_masking(image):
    """배경 무시한 pHash 계산 (에지 기반)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return imagehash.phash(Image.fromarray(edges))


def extract_style_features(image):
    """스타일 특성 추출 (메타 임베딩용)"""
    # 1. 배경색 분포
    bg_color = estimate_background_color(image)
    bg_hsv = cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    
    # 2. 여백 비율
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_gray = np.mean(cv2.cvtColor(bg_color.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY))
    spacing_ratio = np.mean(np.abs(gray - bg_gray) < 20)
    
    # 3. 색상 채도 평균
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_mean = np.mean(hsv[:, :, 1])
    
    # 4. 밝기 평균
    brightness_mean = np.mean(hsv[:, :, 2])
    
    return {
        'bg_hue': float(bg_hsv[0]),
        'bg_saturation': float(bg_hsv[1]),
        'bg_brightness': float(bg_hsv[2]),
        'spacing_ratio': float(spacing_ratio),
        'content_saturation': float(saturation_mean),
        'content_brightness': float(brightness_mean)
    }


# ==================== 3. 전체 전처리 파이프라인 ====================

def preprocess_webtoon_cut(image_path, target_size=320):
    """웹툰 컷 전처리 종합 파이프라인 (한글 경로 대응)"""
    try:
        # 한글 경로 처리: numpy로 읽기
        img_array = np.fromfile(str(image_path), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, None, "Failed to read image"
        
        # 1단계: 과도한 여백/크레딧 컷 필터링
        filtered, spacing_ratio = filter_excessive_spacing(image)
        if filtered is None:
            return None, None, f"Excessive spacing: {spacing_ratio:.2f}"
        
        # 2단계: 상하 극단 여백 제거
        cropped = smart_crop_vertical_spacing(filtered)
        
        # 3단계: 적응형 리사이즈 (배경색 보존)
        resized = adaptive_resize_with_original_bg(cropped, target_size=target_size)
        
        # 4단계: 스타일 특성 추출
        style_features = extract_style_features(resized)
        
        return resized, style_features, "Success"
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def deduplicate_cuts(image_list, hash_threshold=2):
    """pHash 기반 근접중복 제거 (한글 경로 대응)"""
    hashes = {}
    unique_images = []
    duplicates = []
    
    print("\n🔍 중복 제거 중...")
    for img_path in tqdm(image_list, desc="Deduplicating"):
        try:
            # 한글 경로 처리: numpy로 읽기
            img_array = np.fromfile(str(img_path), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"⚠️ 읽기 실패 (건너뜀): {img_path.name}")
                continue
            
            h = perceptual_hash_with_masking(img)
            
            # 기존 해시와 비교
            is_duplicate = False
            for existing_hash, existing_path in hashes.items():
                if h - existing_hash <= hash_threshold:
                    is_duplicate = True
                    duplicates.append((img_path, existing_path))
                    break
            
            if not is_duplicate:
                hashes[h] = img_path
                unique_images.append(img_path)
        
        except Exception as e:
            # 조용히 건너뜀 (너무 많은 경고 방지)
            continue
    
    return unique_images, duplicates


def remove_exact_duplicates(image_list):
    """파일 해시 기반 완전 중복 제거 (빠르고 정확)"""
    print("\n🔍 완전 중복 파일 제거 중...")
    seen_hashes = {}
    unique_images = []
    duplicates = []
    
    for img_path in tqdm(image_list, desc="Checking exact duplicates"):
        try:
            # 파일 내용 해시 계산
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in seen_hashes:
                duplicates.append((img_path, seen_hashes[file_hash]))
            else:
                seen_hashes[file_hash] = img_path
                unique_images.append(img_path)
        except Exception as e:
            # 읽기 실패한 파일은 건너뜀
            continue
    
    print(f"✅ 완전 중복 제거: {len(duplicates)}개 제거, {len(unique_images)}개 유지")
    return unique_images, duplicates


# ==================== 4. 메인 처리 함수 ====================

def process_webtoon_directory(
    input_dir, 
    output_dir, 
    target_size=320, 
    remove_duplicates=False,
    hash_threshold=2,
    remove_exact_duplicates_only=True
):
    """
    웹툰 디렉토리 전체 전처리
    
    Parameters:
    - input_dir: 원본 이미지 디렉토리 경로
    - output_dir: 전처리 결과 저장 경로
    - target_size: 리사이즈 목표 크기 (짧은 변 기준)
    - remove_duplicates: pHash 기반 유사 중복 제거 여부
    - hash_threshold: pHash 해밍 거리 임계값
    - remove_exact_duplicates_only: 완전 중복만 제거 (권장)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 경로 존재 확인
    if not input_path.exists():
        print(f"❌ 입력 경로가 존재하지 않습니다: {input_path}")
        return 0, 0, 0
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 수집
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []
    
    print(f"📂 이미지 파일 수집 중...")
    try:
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    except Exception as e:
        print(f"❌ 파일 수집 중 오류: {e}")
        return 0, 0, 0
    
    print(f"📂 총 {len(image_files)}개 이미지 발견")
    
    if len(image_files) == 0:
        print(f"❌ {input_path}에서 이미지 파일을 찾을 수 없습니다.")
        return 0, 0, 0
    
    # 중복 제거 (선택사항)
    if remove_exact_duplicates_only:
        # 파일 해시 기반 완전 중복만 제거 (권장)
        image_files, duplicates = remove_exact_duplicates(image_files)
    
    if remove_duplicates:
        # pHash 기반 유사 중복 제거 (시간 오래 걸림)
        image_files, duplicates = deduplicate_cuts(image_files, hash_threshold)
        print(f"✅ 유사 중복 제거 완료: {len(duplicates)}개 제거, {len(image_files)}개 유지")
    
    # 전처리 실행
    processed_count = 0
    filtered_count = 0
    error_count = 0
    metadata = {}
    error_log = []  # 에러 로그 추가
    
    print("\n🎨 전처리 시작...")
    for img_path in tqdm(image_files, desc="Processing"):
        # 출력 경로 생성 (원본 디렉토리 구조 유지)
        relative_path = img_path.relative_to(input_path)
        output_file = output_path / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 전처리 실행
        result, style_features, status = preprocess_webtoon_cut(img_path, target_size)
        
        if result is not None:
            # 한글 경로 저장: cv2.imencode 사용
            is_success, buffer = cv2.imencode('.jpg', result)
            if is_success:
                buffer.tofile(str(output_file))
                metadata[str(relative_path)] = {
                    'original_path': str(img_path),
                    'status': status,
                    'style_features': style_features
                }
                processed_count += 1
            else:
                error_count += 1
                error_log.append((str(img_path), "imencode failed"))
        elif "Excessive spacing" in status:
            filtered_count += 1
        else:
            error_count += 1
            error_log.append((str(img_path), status))
    
    # 메타데이터 저장
    metadata_file = output_path / 'preprocessing_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_input': len(image_files),
            'processed': processed_count,
            'filtered': filtered_count,
            'errors': error_count,
            'target_size': target_size,
            'remove_duplicates': remove_duplicates,
            'files': metadata
        }, f, indent=2, ensure_ascii=False)
    
    # 에러 로그 저장
    if len(error_log) > 0:
        error_log_file = output_path / 'error_log.txt'
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"총 {len(error_log)}개 에러\n\n")
            for path, error_msg in error_log[:100]:  # 처음 100개만
                f.write(f"{path}\n  -> {error_msg}\n\n")
        print(f"⚠️  에러 로그: {error_log_file}")
    
    print(f"\n✅ 전처리 완료!")
    print(f"   - 처리 성공: {processed_count}개")
    print(f"   - 필터링됨: {filtered_count}개")
    print(f"   - 오류: {error_count}개")
    print(f"   - 메타데이터: {metadata_file}")
    
    return processed_count, filtered_count, error_count

# ==================== 6. 메인 실행 함수 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='웹툰 컷 이미지 전처리')
    parser.add_argument('--input_dir', type=str, default=r'D:\Crawling\Naver',
                        help='원본 이미지 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default=r'D:\Crawling\Naver_Processed',
                        help='전처리 결과 저장 경로')
    parser.add_argument('--target_size', type=int, default=320,
                        help='리사이즈 목표 크기 (짧은 변 기준, 기본값: 320)')
    parser.add_argument('--remove_duplicates', action='store_true',
                        help='pHash 기반 유사 중복 제거 활성화 (느림)')
    parser.add_argument('--hash_threshold', type=int, default=2,
                        help='pHash 해밍 거리 임계값 (기본값: 2, 낮을수록 엄격)')
    parser.add_argument('--no_exact_duplicate_removal', action='store_true',
                        help='완전 중복 제거 비활성화')
    parser.add_argument('--create_zip', action='store_true',
                        help='결과를 ZIP으로 압축')
    parser.add_argument('--visualize', action='store_true',
                        help='전처리 전후 비교 이미지 생성')
    
    args = parser.parse_args()
    
    # 전처리 실행
    print(f"\n{'='*60}")
    print(f"🎨 웹툰 컷 이미지 전처리 시작")
    print(f"{'='*60}")
    print(f"📂 입력 경로: {args.input_dir}")
    print(f"📂 출력 경로: {args.output_dir}")
    print(f"🎯 목표 크기: {args.target_size}px")
    print(f"🔍 완전 중복 제거: {'비활성화' if args.no_exact_duplicate_removal else '활성화 (권장)'}")
    print(f"🔍 유사 중복 제거: {'활성화' if args.remove_duplicates else '비활성화'}")
    if args.remove_duplicates:
        print(f"   - pHash 임계값: {args.hash_threshold}")
    print(f"{'='*60}\n")
    
    processed, filtered, errors = process_webtoon_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        remove_duplicates=args.remove_duplicates,
        hash_threshold=args.hash_threshold,
        remove_exact_duplicates_only=not args.no_exact_duplicate_removal
    )
    
    print(f"\n{'='*60}")
    print(f"🎉 전처리 완료!")
    print(f"{'='*60}")
    print(f"✅ 처리 성공: {processed}개")
    print(f"🚫 필터링됨: {filtered}개 (과도한 여백)")
    print(f"⚠️  오류: {errors}개")
    print(f"📊 메타데이터: {args.output_dir}/preprocessing_metadata.json")
    print(f"{'='*60}\n")


# ==================== 7. 간단 실행 예시 ====================

"""
터미널에서 실행:

# 1. 기본 실행 (완전 중복만 제거, 권장)
python webtoon_preprocessing.py

# 2. 완전 중복 제거도 끄기 (모든 파일 보존)
python webtoon_preprocessing.py --no_exact_duplicate_removal

# 3. 유사 중복도 제거 (느리지만 더 많이 제거)
python webtoon_preprocessing.py --remove_duplicates --hash_threshold 2

# 4. ZIP + 시각화 추가
python webtoon_preprocessing.py --create_zip --visualize

# 5. 다른 경로 지정
python webtoon_preprocessing.py \
    --input_dir "D:\다른경로\원본" \
    --output_dir "D:\다른경로\처리완료"
"""