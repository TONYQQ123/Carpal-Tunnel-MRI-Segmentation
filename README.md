# 🧠 Carpal Tunnel MRI Segmentation

這是一個基於 **Python** 與 **PyQt5** 開發的圖形化介面 (GUI) 應用程式，專門用於分析手腕 MRI 影像 (T1/T2 weighted images)。

本工具整合了傳統電腦視覺演算法 (OpenCV)，能自動分割並識別以下解剖構造，並計算與 Ground Truth (GT) 的 Dice Coefficient：

1. **CT (Carpal Tunnel)**: 腕隧道區域
2. **FT (Flexor Tendons)**: 屈指肌腱
3. **MN (Median Nerve)**: 正中神經

## ✨ 功能特色 (Features)

* **圖形化操作介面**: 雙欄位設計，左側檢視原始 T1/T2 影像，右側檢視分割結果。
* **自動化預測 (Auto-Prediction)**:
* 整合影像前處理 (CLAHE, Sharpening)。
* 利用形態學與輪廓特徵自動抓取肌腱與神經位置。


* **視覺化比對**:
* 🟢 **綠色遮罩**: Ground Truth (人工標註)。
* 🔴 **紅色框線**: Algorithm Prediction (演算法預測)。


* **數據評估**: 自動計算 **Dice Coefficient**，量化預測準確度。
* **資料導航**: 支援資料夾批次載入，並可透過按鈕或索引跳轉影像。

```

## 🚀 使用說明 (Usage)

### 1. 啟動程式

在終端機執行以下指令開啟主視窗：

```bash
python main.py

```

*(請將 `main.py` 替換成你的檔案名稱)*

### 2. 資料載入 (Load Data)

程式介面分為幾個區塊，請依序載入資料：

1. **載入原始影像**:
* 點擊 `Load T1 folder`: 選擇存放 T1 MRI 影像的資料夾。
* 點擊 `Load T2 folder`: 選擇存放 T2 MRI 影像的資料夾。


2. **載入標註遮罩 (可選)**:
* 若你有 Ground Truth (GT) 遮罩，請點擊下方按鈕分別載入 `CT`, `FT`, `MN` 的 Mask 資料夾。
* 載入後，畫面上會顯示 **綠色** 的 GT 遮罩。



> **注意**: 資料夾內的檔案請確保檔名包含數字順序 (例如 `1.png`, `2.png`...)，程式會依照數字順序自動對齊 T1、T2 與 Mask。

### 3. 執行預測 (Predict)

* 點擊介面下方的 **`Predict`** 按鈕。
* 程式將對所有載入的影像進行運算。
* 運算完成後，畫面上會出現 **紅色框線** 代表預測結果，並顯示 Dice 分數。

### 4. 檢視結果

* **切換分頁**: 右側可切換 `T1` 或 `T2` 分頁，查看該模態下的疊圖效果。
* **切換影像**: 使用左側的 `←` `→` 按鈕或輸入數字索引來切換不同切片 (Slice)。

## 🧠 演算法原理 (Methodology)

本專案不使用深度學習，而是採用傳統影像處理技術進行特徵提取：

1. **前處理 (Preprocessing)**:
* 使用 CLAHE (限制對比度自適應直方圖均衡化) 增強對比。
* 使用 Laplacian Kernel 進行影像銳化。
* Otsu 二值化與形態學運算提取手部輪廓 (ROI)。


2. **肌腱偵測 (FT Segmentation)**:
* 在 **T1** 影像中尋找深色區域 (Dark Regions)。
* 利用 `filter_dark_by_white_center` 過濾掉位於骨頭/脂肪重心上方的非肌腱組織。
* 透過 `keep_clustered_contours` 去除離群雜訊。


3. **腕隧道區域 (CT Segmentation)**:
* 基於偵測到的肌腱區域，計算其凸包 (Convex Hull) 作為腕隧道範圍。


4. **神經偵測 (MN Segmentation)**:
* 在 **T2** 影像中尋找高亮區域 (White Regions)。
* **關鍵邏輯**: 限制搜尋範圍在 **CT 區域的右半部** (`get_right_half_mask`)，因為正中神經通常位於該區域且訊號較強。

## Demo


https://github.com/user-attachments/assets/d6f82920-6b4a-4e62-8aca-9cd33bd4e857
1. **CT Predict**
<img width="184" height="210" alt="1" src="https://github.com/user-attachments/assets/c1d82906-c39f-4f24-98ad-4aa9ab39d992" />

2. **FT Predict**
<img width="189" height="214" alt="2" src="https://github.com/user-attachments/assets/dfe03fe1-e299-4df0-af7e-e405351f4475" />

3. **MN Predict**
<img width="188" height="211" alt="3" src="https://github.com/user-attachments/assets/3b78765e-bdf5-44a2-a149-e24c5d39471d" />



