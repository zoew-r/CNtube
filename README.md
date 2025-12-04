# CNtube - YouTube 中文語言學習助手

CNtube 是一個結合 YouTube 影片轉錄與 AI 語意分析的語言學習工具。專為學習台灣華語（Traditional Chinese, Taiwan usage）的學生設計。

## 功能特色

- **即時轉錄**：輸入 YouTube 網址，即時生成逐字稿。
- **同步字幕**：逐字稿會隨著影片播放進度同步顯示。
- **AI 語意分析**：提供單詞的拼音、注音、英文解釋、文法說明及邏輯分析。
- **台灣在地化**：針對台灣華語用語習慣進行優化，並提供注音符號 (Bopomofo)。

## 系統需求

- **Python 3.10+**
- **FFmpeg** (用於音訊處理)

### 安裝 FFmpeg
- **Mac (使用 Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Windows**:
  請下載 FFmpeg 並將其加入系統環境變數 PATH 中。

## 快速開始 (Mac/Linux)

我們提供了一個自動化腳本，可以一鍵完成安裝並啟動程式。

1. 下載專案後，開啟終端機 (Terminal) 並進入專案資料夾。
2. 執行以下指令：
   ```bash
   bash run.sh
   ```
   
   這個腳本會自動：
   - 檢查並建立 Python 虛擬環境 (.venv)
   - 安裝所有必要的套件 (requirements.txt)
   - 啟動網頁伺服器

3. 程式啟動後，請在瀏覽器打開：`http://localhost:5001`

## 手動安裝

如果您無法使用 `run.sh`，也可以手動安裝：

1. **建立虛擬環境**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate   # Windows
   ```

2. **安裝套件**
   ```bash
   pip install -r requirements.txt
   ```

3. **設定環境變數**
   請建立一個 `.env` 檔案，並填入您的 API 金鑰 (如果使用 OpenAI)：
   ```
   OPENAI_API_KEY=your_api_key_here
   # 或者使用 Ollama (預設)
   USE_OLLAMA=true
   ```

4. **啟動程式**
   ```bash
   python -m services.app
   ```

## 使用說明

1. **取得逐字稿**：在上方輸入框貼上 YouTube 影片網址，點擊「取得逐字稿」。
2. **觀看影片**：影片載入後，下方的字幕區塊會隨著影片同步顯示目前的句子。
3. **語意分析**：複製右側逐字稿中感興趣的句子，貼到下方的分析區塊，點擊「開始分析」。
