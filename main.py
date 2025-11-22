import requests
from bs4 import BeautifulSoup
import os

headers = {
    "User-Agent": "Mozilla/5.0"
}

title_id = "758037"  
total_episodes = 210

save_dir = r"D:\Crawling\참교육"
os.makedirs(save_dir, exist_ok=True)

for episode_no in range(18, total_episodes + 1):
    url = f"https://comic.naver.com/webtoon/detail?titleId={title_id}&no={episode_no}"
    res = requests.get(url, headers=headers)
    res = requests.get(url, headers=headers)
    res.raise_for_status() 
    soup = BeautifulSoup(res.text, "html.parser")

    img_tags = soup.select("div.wt_viewer img")  

    print(f"이미지 개수: {len(img_tags)}")

    for i, img in enumerate(img_tags, start=1):
        img_url = img["src"]
        img_data = requests.get(img_url, headers=headers).content
        filename = os.path.join(save_dir, f"{episode_no}_{i}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)

    print(f"{episode_no}화 이미지 저장 완료 ({len(img_tags)}컷)")