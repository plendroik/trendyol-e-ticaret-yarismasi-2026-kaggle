import os, urllib.request, json

def test_api():
    kf = r"c:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\openai_key.txt"
    key = open(kf, encoding="utf-8").read().strip()
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Hi, answer with 1 word."}
        ],
        "temperature": 0.0,
        "max_tokens": 5
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode("utf-8"))
            print("Response:", res["choices"][0]["message"]["content"])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_api()
