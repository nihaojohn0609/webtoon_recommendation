import os
import csv

# 기준 폴더 경로
base_dir = r"D:\Crawling\Naver"

# 결과 저장용 리스트
data = []

# 1단계: 웹툰별 폴더 순회
for webtoon_name in os.listdir(base_dir):
    webtoon_path = os.path.join(base_dir, webtoon_name)
    if not os.path.isdir(webtoon_path):
        continue

    # 2단계: 회차(bundle_01 등) 폴더 순회
    for episode_name in os.listdir(webtoon_path):
        episode_path = os.path.join(webtoon_path, episode_name)
        if not os.path.isdir(episode_path):
            continue

        # 3단계: jpg 파일 개수 세기
        jpg_count = sum(1 for f in os.listdir(episode_path)
                        if f.lower().endswith(".jpg"))

        # 결과 저장
        data.append([webtoon_name, episode_name, jpg_count])

# 결과 CSV로 저장
output_csv = "webtoon_counts.csv" 
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["웹툰명", "회차", "이미지_개수"])
    writer.writerows(data)

print(f"✅ 완료: {output_csv} 파일이 생성되었습니다.")
