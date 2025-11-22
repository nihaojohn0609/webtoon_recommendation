import os
import json
import time
import random
from bs4 import BeautifulSoup

# Selenium 관련 라이브러리
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------
# 설정
# ---------------------------
BASE_DIR = r"D:\Crawling\Naver_Processed"
OUTPUT_JSON = "webtoon_metadata.json"
FAILED_TITLES_JSON = "failed_titles.json"

# 공식 장르 기준 (하드코딩)
OFFICIAL_GENRES = {
    "일상", "개그", "판타지", "액션", "드라마", "순정", "감성", "스릴러", 
    "무협/사극", "스포츠", "로맨스", "학원", "공포", "미스터리", "시대극",
    "BL", "GL", "옴니버스", "에피소드", "스토리", "로판", "무협", "사극", "성인"
}

# ---------------------------
# 브라우저 설정 함수
# ---------------------------
def create_driver():
    chrome_options = Options()
    # headless를 True로 하면 브라우저 창이 안 뜨고 백그라운드에서 돕니다. (속도 빠름)
    # 처음엔 잘 되는지 보려면 False로 두세요.
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    
    # 봇 탐지 회피 옵션
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# ---------------------------
# 데이터 추출 함수 (Selenium 이용)
# ---------------------------
def scrape_with_selenium(driver, title_id, folder_title):
    url = f"https://comic.naver.com/webtoon/list?titleId={title_id}"
    
    metadata = {
        "title": folder_title, # 기본값으로 폴더명 사용
        "titleId": title_id,
        "writer": "",
        "genre": [],
        "keywords": [],
        "summary": "",
        "status": ""
    }

    try:
        # 1. 브라우저로 이동
        driver.get(url)
        
        # 2. 페이지 로딩 대기 (안전하게 2~3초 대기)
        # 네이버 웹툰은 동적 로딩이라 시간이 조금 필요합니다.
        time.sleep(random.uniform(2, 3))
        
        # 3. 현재 브라우저에 보이는 HTML 소스 가져오기
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        # -------------------------------------------------
        # 데이터 추출 (HTML 구조 분석)
        # -------------------------------------------------
        
        # [제목] Meta 태그가 가장 정확함
        og_title = soup.select_one('meta[property="og:title"]')
        if og_title:
            metadata["title"] = og_title.get("content", "").strip()

        # [요약]
        og_desc = soup.select_one('meta[property="og:description"]')
        if og_desc:
            metadata["summary"] = og_desc.get("content", "").strip()

        # [작가] 'ContentMetaInfo__category' 클래스가 포함된 태그 찾기
        # (클래스 이름이 일부 바뀌어도 찾을 수 있게 부분 매칭 사용)
        writers = []
        author_tags = soup.find_all(class_=lambda c: c and 'ContentMetaInfo__category' in c)
        for tag in author_tags:
            text = tag.get_text(strip=True)
            # 글, 그림, 원작 등의 텍스트가 있으면 작가 이름으로 간주
            if any(x in text for x in ['글', '그림', '원작']):
                clean_name = text.replace('글', '').replace('그림', '').replace('원작', '').strip()
                if clean_name:
                    writers.append(clean_name)
        
        # 중복 제거 후 저장
        metadata["writer"] = ', '.join(sorted(set(writers))) if writers else ""

        # [상태] (연재중/완결 등)
        status_tag = soup.find(class_=lambda c: c and 'ContentMetaInfo__info_item' in c)
        if status_tag:
            text = status_tag.get_text(strip=True)
            metadata["status"] = text.split('∙')[0].strip() if '∙' in text else text

        # [태그 & 장르] 버튼이나 링크 중 #으로 시작하는 것 모두 수집
        # 특정 div 안을 찾지 않고 전체에서 찾음 (구조 변경 방어)
        all_buttons = soup.find_all(['a', 'button'])
        
        for tag in all_buttons:
            text = tag.get_text(strip=True)
            if text.startswith('#') and len(text) > 1:
                clean_tag = text.replace('#', '').strip()
                
                # 장르/키워드 분류
                if clean_tag in OFFICIAL_GENRES:
                    if clean_tag not in metadata["genre"]:
                        metadata["genre"].append(clean_tag)
                else:
                    if clean_tag not in metadata["keywords"]:
                        metadata["keywords"].append(clean_tag)

        return metadata

    except Exception as e:
        print(f"[ERROR] Selenium 처리 중 오류 (ID={title_id}): {e}")
        return None

# ---------------------------
# 메인 실행
# ---------------------------
def main():
    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] 폴더 경로 확인 필요: {BASE_DIR}")
        return

    # 폴더 목록에서 ID 추출
    all_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    target_items = []
    for d in all_dirs:
        parts = d.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            target_items.append({"titleId": parts[0], "title": parts[1]})
    
    print(f"총 {len(target_items)}개의 작품을 처리합니다. (브라우저 실행 중...)")
    
    # 브라우저 시작
    driver = create_driver()
    
    metadata_list = []
    failed_list = []

    try:
        for idx, item in enumerate(target_items, 1):
            tid = item['titleId']
            folder_title = item['title']
            
            print(f"[{idx}/{len(target_items)}] '{folder_title}' ({tid}) 접속 중...", end=" ")
            
            result = scrape_with_selenium(driver, tid, folder_title)
            
            if result:
                # 데이터 확인
                has_genre = len(result['genre']) > 0
                has_keyword = len(result['keywords']) > 0
                
                if has_genre or has_keyword:
                    print(f"✅ [성공] 장르: {result['genre']} | 키워드: {len(result['keywords'])}개")
                else:
                    print(f"⚠️ [주의] 태그 없음 (페이지 확인 필요)")
                
                metadata_list.append(result)
            else:
                print(f"❌ [실패]")
                failed_list.append(item)
            
            # 너무 빠르면 차단될 수 있으니 약간 대기
            # time.sleep(1) 

    except KeyboardInterrupt:
        print("\n[중단] 사용자에 의해 중단되었습니다. 현재까지의 데이터를 저장합니다.")
        
    finally:
        # 브라우저 종료
        driver.quit()
        
        # 저장
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        print(f"\n💾 최종 저장 완료: {OUTPUT_JSON}")
        
        if failed_list:
            with open(FAILED_TITLES_JSON, "w", encoding="utf-8") as f:
                json.dump(failed_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()