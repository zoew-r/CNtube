import requests
from bs4 import BeautifulSoup
import time # 引入 time 函式庫用於設定延遲

# --- 全域設定 ---
OUTPUT_FILENAME = 'grammar_corpus.txt'

# 模擬瀏覽器行為，避免被網站阻擋
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print(f"--- 開始批次爬取 {OUTPUT_FILENAME} ---")

# 使用寫入模式 'w' 先清空/創建檔案，確保每次執行都是從頭開始追加
# 注意：如果希望保留以前的內容，請將 'w' 改為 'a'
with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
    f.write(f"網站爬取結果開始於：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n" + "="*50 + "\n\n")

levels = [1, 8]

for level in levels:
    for page in range(1, 14):
        
        # --- 1. 設定目標 (網址參數在迴圈內動態變化) ---
        # U+00A0 字符已移除
        URL = f'https://coct.naer.edu.tw/grammar.jsp?lv={level}&pgSize=20&page={page}' 
        
        print(f"正在爬取 Level: {level}, Page: {page} -> {URL}")
        
        try:
            # --- 2. 獲取網頁內容 ---
            response = requests.get(URL, headers=HEADERS, timeout=10)
            response.raise_for_status() 

            # 設定編碼，避免中文亂碼
            response.encoding = response.apparent_encoding 
            
            # 使用 BeautifulSoup 解析 HTML 內容
            soup = BeautifulSoup(response.text, 'html.parser')

            # --- 3. 提取所有可見文字 ---
            # sep='\n\n' 確保不同區塊的文字之間會有空行
            text_content = soup.get_text(separator='\n\n', strip=True)

            # 為內容添加標記，方便區分每一頁的內容
            formatted_content = f"\n\n--- Start of LV{level} PAGE{page} ---\n\n"
            formatted_content += text_content
            formatted_content += f"\n\n--- End of LV{level} PAGE{page} ---\n\n"

            # --- 4. 儲存到 TXT 檔案 (使用追加模式 'a') ---
            with open(OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
                f.write(formatted_content)

            print(f"✅ Level: {level}, Page: {page} 內容已追加儲存。")
            
            # 💡 最佳實踐：加入延遲，避免短時間內對網站發送過多請求
            time.sleep(1) # 每次請求間隔 1 秒

        except requests.exceptions.HTTPError as http_err:
            print(f"❌ 發生 HTTP 錯誤 (LV{level}, P{page}): {http_err} - 該頁面可能不存在。")
        except requests.exceptions.RequestException as e:
            print(f"❌ 請求失敗或連線錯誤 (LV{level}, P{page}): {e}")
        except Exception as e:
            print(f"❌ 處理過程中發生未知錯誤 (LV{level}, P{page}): {e}")

print("\n" + "="*50)
print(f"🎉 批次爬取任務完成！所有內容已追加儲存至 {OUTPUT_FILENAME}")
print("="*50)