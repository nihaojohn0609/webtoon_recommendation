import os
import re
import random

# --- 설정 ---
ROOT_DIR = r"D:\Crawling\Naver" 
SAMPLE_RATE = 0.1  # 10%만 남기기 (1/10 샘플링)

# 층화 샘플링 가중치 설정
SECTION_WEIGHTS = {
    'START': 0.2,   # 초반 20%
    'MIDDLE': 0.6,  # 중반 60%
    'END': 0.2      # 종단 20%
}

random.seed(42)  # 재현 가능하도록 시드 설정

if not os.path.isdir(ROOT_DIR):
    print(f"❌ FATAL ERROR: 설정된 ROOT_DIR 경로가 유효하지 않거나 존재하지 않습니다.")
    print(f"   경로를 확인해주세요: {ROOT_DIR}")
    exit()

print(f"[{ROOT_DIR}] 폴더 내의 모든 웹툰 이미지에 대해 1/10 층화 샘플링을 적용합니다.")
print(f"가중치: 초반 {SECTION_WEIGHTS['START']}, 중반 {SECTION_WEIGHTS['MIDDLE']}, 종단 {SECTION_WEIGHTS['END']}")
print(f"⚠️  각 bundle 폴더에는 최소 10개의 컷이 남습니다.")
print("-" * 50)

deleted_count = 0
kept_count = 0
error_count = 0
bundle_count = 0

for root, dirs, files in os.walk(ROOT_DIR):
    # bundle_XX 폴더만 처리
    if not re.search(r'bundle_\d+', root):
        continue
    
    bundle_count += 1
    bundle_name = os.path.basename(root)
    
    # jpg 파일만 수집
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    
    if not jpg_files:
        print(f"  ⚠️ {bundle_name}: 이미지 파일 없음")
        continue
    
    # 파일명에서 에피소드 번호와 컷 번호 추출하여 정렬
    file_info_list = []
    for f in jpg_files:
        try:
            # 파일명 형식: {episode_no}_{cut_no}.jpg
            parts = f.split('_')
            if len(parts) >= 2:
                episode_no = int(parts[0])
                cut_no = int(parts[1].split('.')[0])
                file_info_list.append({
                    'name': f,
                    'episode': episode_no,
                    'cut': cut_no
                })
        except ValueError:
            print(f"  ⚠️ {bundle_name}: 파일명 파싱 실패 - {f}")
            continue
    
    if not file_info_list:
        continue
    
    # 컷 번호 기준으로 정렬
    file_info_list.sort(key=lambda x: (x['episode'], x['cut']))
    
    total_cuts = len(file_info_list)
    
    # ✅ 최소 10컷 이상 유지
    keep_count_target = max(10, int(total_cuts * SAMPLE_RATE))
    
    # 층화 샘플링: 초반, 중반, 종단으로 나누기
    start_end_idx = max(1, int(total_cuts * 0.2))  # 초반 20%
    end_start_idx = max(start_end_idx + 1, int(total_cuts * 0.8))  # 종단 20% 시작점
    
    start_section = list(range(0, start_end_idx))
    middle_section = list(range(start_end_idx, end_start_idx))
    end_section = list(range(end_start_idx, total_cuts))
    
    # 각 섹션에서 선택할 개수 계산
    start_keep = max(0, round(keep_count_target * SECTION_WEIGHTS['START']))
    middle_keep = max(0, round(keep_count_target * SECTION_WEIGHTS['MIDDLE']))
    end_keep = max(0, round(keep_count_target * SECTION_WEIGHTS['END']))
    
    # 반올림 오차 보정
    total_allocated = start_keep + middle_keep + end_keep
    if total_allocated < keep_count_target:
        middle_keep += (keep_count_target - total_allocated)
    elif total_allocated > keep_count_target:
        middle_keep = max(0, middle_keep - (total_allocated - keep_count_target))
    
    keep_indices = set()
    
    # 각 섹션에서 랜덤 샘플링
    if start_section and start_keep > 0:
        actual_start_keep = min(start_keep, len(start_section))
        keep_indices.update(random.sample(start_section, actual_start_keep))
    
    if middle_section and middle_keep > 0:
        actual_middle_keep = min(middle_keep, len(middle_section))
        keep_indices.update(random.sample(middle_section, actual_middle_keep))
    
    if end_section and end_keep > 0:
        actual_end_keep = min(end_keep, len(end_section))
        keep_indices.update(random.sample(end_section, actual_end_keep))
    
    # 혹시 부족하면 전체에서 추가 샘플링 (최소 10개 보장)
    if len(keep_indices) < 10:
        additional_needed = 10 - len(keep_indices)
        available = set(range(total_cuts)) - keep_indices
        if available:
            additional = min(additional_needed, len(available))
            keep_indices.update(random.sample(list(available), additional))
    elif len(keep_indices) < keep_count_target:
        remaining = keep_count_target - len(keep_indices)
        available = set(range(total_cuts)) - keep_indices
        if available:
            additional = min(remaining, len(available))
            keep_indices.update(random.sample(list(available), additional))
    
    # 파일 삭제 처리
    bundle_deleted = 0
    bundle_kept = 0
    
    for idx, file_info in enumerate(file_info_list):
        try:
            file_path = os.path.join(root, file_info['name'])
            
            # 선택된 인덱스면 유지
            if idx in keep_indices:
                kept_count += 1
                bundle_kept += 1
                continue
            
            # 나머지는 삭제
            if not os.path.exists(file_path):
                continue
            
            # Windows 긴 경로 지원
            if len(file_path) > 260 and not file_path.startswith('\\\\?\\'):
                file_path = '\\\\?\\' + os.path.abspath(file_path)
            
            os.chmod(file_path, 0o777)
            os.remove(file_path)
            deleted_count += 1
            bundle_deleted += 1
                
        except PermissionError:
            error_count += 1
            print(f"  ❌ {bundle_name}: 권한 오류 - {file_info['name']}")
        except Exception as e:
            error_count += 1
            print(f"  ❌ {bundle_name}: 삭제 실패 - {file_info['name']} ({type(e).__name__}: {e})")
    
    # 진행상황 표시
    print(f"  ✅ {bundle_name}: {total_cuts}컷 → {bundle_kept}컷 유지 ({bundle_deleted}개 삭제)")
    
    # 주기적으로 전체 진행상황 표시
    if bundle_count % 50 == 0:
        print(f"\n  📊 중간 집계: {bundle_count}개 bundle 처리 완료")
        print(f"     삭제: {deleted_count}개 | 유지: {kept_count}개 | 오류: {error_count}개\n")

print("-" * 50)
print(f"✅ 층화 랜덤 샘플링 완료")
print(f"📦 처리된 bundle 폴더 수: {bundle_count}개")
print(f"🗑️ 삭제된 파일 수: {deleted_count}개")
print(f"📁 유지된 파일 수: {kept_count}개")
print(f"⚠️ 오류 발생 파일 수: {error_count}개")
if kept_count + deleted_count > 0:
    print(f"📊 최종 보존율: {kept_count / (kept_count + deleted_count) * 100:.1f}%")
print("-" * 50)
