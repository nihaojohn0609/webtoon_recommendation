import requests
from bs4 import BeautifulSoup
import re
import os
import random
import time
import unicodedata
import json 
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# ----------------------------------------------------------------------
# 🌟 설정 및 전역 변수
# ----------------------------------------------------------------------

def safe_filename(name: str, max_length: int = 80) -> str:
    """
    Windows에서 안전하게 사용할 수 있는 폴더 이름으로 변환.
    - 금지문자: \ / : * ? " < > | , 등
    - 이모지, 비표준 문자 제거
    - 너무 긴 이름은 자름
    """
    # 금지 문자 → '_'
    name = re.sub(r'[\\/:*?"<>|,]', '_', name)

    # 이모지/비표준 문자 제거
    name = ''.join(c for c in name if unicodedata.category(c)[0] != 'So')

    # 공백 정리
    name = name.strip()

    # 너무 길면 잘라내기
    if len(name) > max_length:
        name = name[:max_length].rstrip('_ ')

    return name

# 🌟 최신 10화는 유료/미리보기일 확률이 높으므로 샘플링에서 제외합니다.
EXCLUDE_EPISODES = 10 
# 동시에 실행할 최대 스레드 개수 (안정화를 위해 8개 유지)
MAX_WORKERS = 8 

COOKIES = {
    # 🚨 여기에 유효한 NID_AUT 값을 입력하세요. (최신 획득 권장)
    "NID_AUT": "8oSmn2jYq1DUDmEB13/YTwAxbE1B8Hlh2LU1addkE1+n3s0XhZKi7Ccr3nn5PvcA", 
    # 🚨 여기에 유효한 NID_SES 값을 입력하세요. (최신 획득 권장)
    "NID_SES": "AAABn4KdVEaRIkWQpXjrbM3FpFA0hKQsZfV7EAjzgom5UGrC5dzEtF/B7m31gmyBwyC3pABqFlMZoCy/dSujLEMaef8RRqG50cn471msGfe3SOldTdDZMq/Q+N6/YaMPV+bIsWPn6TmuZ7CvXynRihptN2U9C3kDVJWX+lftelRPISs4WZ6MS+l+DODOVeRIp3gQE3PIcBzccdjkMHa510tlTMGFQIrt5pieQEDTw0cgpMSXzsnjweHEhQh7+zYswNeaae4WHLdf6gMIaVIGIYoqUqfGMiWtFlYwAvPlliUMmcrznlNTaRIkdHlyaxDjx+aYB9wM3JIeat7bNQUTXLID5M4Eg2j0m4iefpSD9W11KMUIVEyg8RN614wyUC6MYMOmVKPZL4rGMQIfxIp48Yqy4kpZrIdkinZyOaKIuHsAtNsLW3TWV+dHdk19kQN5HMvY+6u8E9KFRfkRmsOW3jX0LTpY7kS7er3nFKSO7QyBNopFR+foDnrsnMob87d7/ojaQxr2vkRjvXyVYojWK/Rrqkp9viBjppXNruddCIeP22vI",
}

USE_LOGIN = bool(COOKIES.get("NID_AUT") and COOKIES.get("NID_SES"))

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://comic.naver.com/"
}

root_dir = r"D:\Crawling\Naver"
os.makedirs(root_dir, exist_ok=True)

ADULT_WEBTOON_OVERRIDES = {
    "839004": 120, "833620": 90, "842079": 55, "833052": 78, 
}

print_lock = Lock()


# ----------------------------------------------------------------------
# 🌟 요청 함수 (Session 사용) - 변경 없음
# ----------------------------------------------------------------------

main_session = requests.Session()
main_session.headers.update(headers)
if USE_LOGIN:
    main_session.cookies.update(COOKIES)

def create_thread_session():
    s = requests.Session()
    s.headers.update(headers)
    if USE_LOGIN:
        s.cookies.update(COOKIES)
    return s

def download_image_with_retry(session, img_url, file_path, retries=3, delay=0.5):
    for i in range(retries):
        try:
            res = session.get(img_url, timeout=10) 
            res.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(res.content)
            return True
        except Exception:
            time.sleep(delay)
    return False

def get_total_episodes(title_id):
    if not USE_LOGIN and title_id in ADULT_WEBTOON_OVERRIDES:
        return ADULT_WEBTOON_OVERRIDES[title_id]
    try:
        api_url = f"https://comic.naver.com/api/article/list?titleId={title_id}&page=1&sort=ASC"
        res = main_session.get(api_url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "totalCount" in data:
            return data["totalCount"]
    except Exception:
        pass
    try:
        list_url = f"https://comic.naver.com/webtoon/list?titleId={title_id}"
        res2 = main_session.get(list_url)
        res2.raise_for_status()
        soup = BeautifulSoup(res2.text, "html.parser")
        episode_count = soup.select_one("span.total")
        if episode_count:
            match = re.search(r'(\d+)', episode_count.text)
            if match:
                return int(match.group(1))
        latest_episode_link = soup.select_one("td.title a")
        if latest_episode_link:
            href = latest_episode_link.get('href', '')
            match = re.search(r'no=(\d+)', href)
            if match:
                return int(match.group(1))
    except Exception:
        pass
    return 0

# ----------------------------------------------------------------------
# 🌟 스레드 실행 함수 - 변경 없음
# ----------------------------------------------------------------------

def download_episode(w, episode_no, bundle_dir):
    session = create_thread_session()
    
    try:
        url = f"https://comic.naver.com/webtoon/detail?titleId={w['title_id']}&no={episode_no}"
        res = session.get(url, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        img_tags = soup.select("div.wt_viewer img")
        
        if not img_tags:
            log_message = f"    ⚠️ {episode_no}화 이미지 0컷: 접근 권한 문제 (쿠키 만료/유료 회차) 예상."
            with print_lock:
                print(log_message)
            return f"Fail: {episode_no} (0 images)"

        
        for i, img in enumerate(img_tags, start=1):
            img_url = img["src"]
            fname = os.path.join(bundle_dir, f"{episode_no}_{i}.jpg")
            
            if not download_image_with_retry(session, img_url, fname):
                 log_message = f"    ❌ {episode_no}화 이미지 {i} 다운로드 실패."
                 with print_lock:
                    print(log_message)
                    
        log_message = f"    ▶ {episode_no}화 이미지 {len(img_tags)}컷 완료"
        with print_lock:
            print(log_message)
            
        return f"Success: {episode_no}"

    except Exception as e:
        log_message = f"    ❌ {episode_no}화 오류: {type(e).__name__}: {e}"
        with print_lock:
            print(log_message)
        return f"Error: {episode_no}"
        
    finally:
        time.sleep(4) 


# ----------------------------------------------------------------------
# 🌟 메인 로직 (샘플링 로직 수정)
# ----------------------------------------------------------------------

# ... (웹툰 목록 탐색 부분은 변경 없음)

if USE_LOGIN:
    print("🔐 로그인 쿠키가 설정되었습니다. 성인/유료 웹툰 접근 가능!")
else:
    print("⚠️  로그인 쿠키가 없습니다. 일반 웹툰만 다운로드됩니다.")
    print("   성인 웹툰 다운로드를 원하시면 코드 상단에 NID_AUT, NID_SES 쿠키를 입력하세요.\n")

api_url = "https://comic.naver.com/api/webtoon/titlelist/weekday"
res = main_session.get(api_url) 
res.raise_for_status()
data = res.json()["titleListMap"]

webtoons = []

for day, items in data.items():
    print(f"\n{day.upper()} 요일 웹툰 탐색 중... ({len(items)}개)")
    for item in items:
        title = item["titleName"]
        title_id = item["titleId"]

        total_episodes = get_total_episodes(title_id)
        
        status = "🔞" if (not USE_LOGIN and title_id in ADULT_WEBTOON_OVERRIDES) else "✅"
        print(f"  {status} {title} (titleId={title_id}, 총 {total_episodes}화)")

        if total_episodes >= 50:
            webtoons.append({
                "day": day, "title": title, "title_id": title_id, "total_episodes": total_episodes
            })

        time.sleep(0.1) 

print(f"\n✅ 50화 이상 웹툰 {len(webtoons)}개 발견")

# 🎲 연속 5화 & 비중복 10세트 샘플링 로직 (수정 적용)
random.seed(42)

total_start_time = time.time()

for w in webtoons:
    total_episodes = w["total_episodes"]
    
    if total_episodes < 50:
        continue

    # 🌟 샘플링 대상 회차 수 조정
    sampling_end_episodes = total_episodes - EXCLUDE_EPISODES
    
    # 제외 후에도 50화 미만이면 건너뛰기
    if sampling_end_episodes < 50: 
        print(f"  Skipping {w['title']}: {total_episodes}화 중 최신 {EXCLUDE_EPISODES}화 제외 시 50화 미만")
        continue

    title_dir = os.path.join(root_dir, f"{w['day']}_{safe_filename(w['title'])}")
    os.makedirs(title_dir, exist_ok=True)

    # 🌟 조정된 회차 수를 기준으로 묶음 개수 계산
    max_bundle_count = sampling_end_episodes // 3
    num_bundles_to_select = min(5, max_bundle_count) 

    possible_bundle_indices = list(range(max_bundle_count))
    random_bundle_indices = random.sample(possible_bundle_indices, num_bundles_to_select)
    random_bundle_indices.sort()

    final_bundles = []
    
    for index in random_bundle_indices:
        # 회차 번호는 항상 1부터 시작 (1, 6, 11, ...)
        start_ep = index * 3 + 1
        bundle = list(range(start_ep, start_ep + 3))
        final_bundles.append(bundle)

    total_sampled_episodes = len(final_bundles) * 3
    # 로그 출력 시 제외 회차 정보 추가
    print(f"\n🎨 {w['title']} ({w['day']}) → 총 {total_episodes}화 중 최신 {EXCLUDE_EPISODES}화 제외 후 {total_sampled_episodes}화 샘플링 시작 ({len(final_bundles)}개 묶음)")

    # ------------------------------------------------------------------
    # 🚀 멀티스레딩 적용 (변경 없음)
    # ------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_episode = {}
        
        for b_idx, bundle in enumerate(final_bundles, 1):
            bundle_dir = os.path.join(title_dir, f"bundle_{b_idx:02d}")
            os.makedirs(bundle_dir, exist_ok=True)
            print(f"  📦 묶음 {b_idx} ({bundle}) (병렬 처리 시작)")

            for episode_no in bundle:
                future = executor.submit(download_episode, w, episode_no, bundle_dir)
                future_to_episode[future] = episode_no

        for future in future_to_episode:
            future.result() 

total_end_time = time.time()
print(f"\n✅ 전체 다운로드 완료! (총 소요 시간: {total_end_time - total_start_time:.2f}초)")