import requests

url = "https://webimg7.com/202510/25/202510250037137854481.jpg"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.webtoons.com/en/canvas/some-webtoon/viewer?episode_no=31"  # ← 실제 페이지 URL
}

res = requests.get(url, headers=headers)

if res.status_code == 200:
    with open("image.jpg", "wb") as f:
        f.write(res.content)
    print("✅ 다운로드 성공")
else:
    print("❌ 접근 실패:", res.status_code)
