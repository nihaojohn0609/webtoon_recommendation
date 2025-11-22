import os
import re
import time

# --- 설정 ---
ROOT_DIR = r"D:\Crawling\Naver" 
CUTS_TO_REMOVE_START = 5 # 앞에서 제거할 컷 수
CUTS_TO_REMOVE_END = 5   # 뒤에서 제거할 컷 수

if not os.path.isdir(ROOT_DIR):
    print(f"❌ FATAL ERROR: 설정된 ROOT_DIR 경로가 유효하지 않거나 존재하지 않습니다.")
    print(f"   경로를 확인해주세요: {ROOT_DIR}")
    exit()

deleted_count = 0
error_count = 0

for root, dirs, files in os.walk(ROOT_DIR):
    if not re.search(r'bundle_\d+', root):
        continue

    # 에피소드별로 그룹화
    episodes = {}
    
    for f in files:
        if f.endswith('.jpg'):
            try:
                parts = f.split('_')
                if len(parts) >= 2:
                    episode_no = int(parts[0])
                    cut_no = int(parts[1].split('.')[0])
                    
                    if episode_no not in episodes:
                        episodes[episode_no] = []
                    
                    episodes[episode_no].append({'name': f, 'cut': cut_no})
            except ValueError:
                continue

    # 각 에피소드별로 처리
    for episode_no, episode_files in episodes.items():
        if not episode_files:
            continue
        
        # 컷 번호 기준 오름차순 정렬
        episode_files.sort(key=lambda x: x['cut'])
        
        total_cuts = len(episode_files)
        files_to_delete = []

        # 1. 초반 컷 삭제 대상 선정
        for i in range(min(CUTS_TO_REMOVE_START, total_cuts)):
            files_to_delete.append(episode_files[i]['name'])

        # 2. 종단 컷 삭제 대상 선정
        if total_cuts > CUTS_TO_REMOVE_START + CUTS_TO_REMOVE_END:
            start_index = total_cuts - CUTS_TO_REMOVE_END
            for i in range(start_index, total_cuts):
                files_to_delete.append(episode_files[i]['name'])
        elif total_cuts > CUTS_TO_REMOVE_START:
            remaining_cuts = total_cuts - CUTS_TO_REMOVE_START
            cuts_to_remove_from_end = min(CUTS_TO_REMOVE_END, remaining_cuts)
            start_index = total_cuts - cuts_to_remove_from_end
            for i in range(start_index, total_cuts):
                files_to_delete.append(episode_files[i]['name'])
        
        # 3. 중복 제거 후 실제 파일 삭제
        unique_files_to_delete = set(files_to_delete)
        
        for filename in unique_files_to_delete:
            file_path = os.path.join(root, filename)
            try:
                # 파일이 존재하는지 확인
                if not os.path.exists(file_path):
                    print(f"  [SKIP] 파일이 존재하지 않음: {file_path}")
                    continue
                
                # 읽기 전용 속성 제거 (Windows)
                os.chmod(file_path, 0o777)
                
                # 파일 삭제
                os.remove(file_path)
                deleted_count += 1
                
                # 진행상황 표시 (100개마다)
                if deleted_count % 100 == 0:
                    print(f"  진행중... {deleted_count}개 삭제됨")
                    
            except PermissionError as e:
                error_count += 1
                print(f"  [ERROR] 권한 오류: {file_path}")
                print(f"         {e}")
            except OSError as e:
                error_count += 1
                print(f"  [ERROR] 파일 삭제 실패: {file_path}")
                print(f"         {e}")
            except Exception as e:
                error_count += 1
                print(f"  [ERROR] 예상치 못한 오류: {file_path}")
                print(f"         {e}")

print("-" * 50)
print(f"✅ 1단계 완료: 초반 {CUTS_TO_REMOVE_START}컷, 종단 {CUTS_TO_REMOVE_END}컷 제거.")
print(f"🗑️ 총 삭제된 파일 수: {deleted_count}개")
print(f"⚠️ 오류 발생 파일 수: {error_count}개")
print("-" * 50)