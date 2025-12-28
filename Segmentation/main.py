import sys
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QGroupBox, QHBoxLayout, QVBoxLayout, QSpinBox, QTabWidget, QGridLayout,
    QFrame, QMessageBox
)
    
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
import numpy as np

SIZE = 350

def make_image_box(size=SIZE):
    lbl = QLabel()
    lbl.setFixedSize(size, size)
    lbl.setFrameStyle(QFrame.Box | QFrame.Plain)
    lbl.setLineWidth(3)
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


def draw_mask(image, mask):
    """
    image: RGB uint8 (H, W, 3)
    mask: 0/1 或 bool (H, W)
    回傳：亮綠半透明 overlay 後的 RGB uint8
    """
    masked_image = image.copy()
    mask_bool = mask.astype(bool)

    # 將 mask 區域塗成亮綠色
    masked_image[mask_bool] = (0, 255, 0)

    # blend: 原圖 30% + 綠色 70%
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def draw_predict_mask(base_img, gt_mask, pred_mask):
    """
    base_img: RGB uint8 (H, W, 3)
    gt_mask: 0/1 GT mask
    pred_mask: 0/1 預測 mask（畫紅線）
    """
    # Step 1: 先畫綠色透明 GT mask
    overlay = draw_mask(base_img, gt_mask)

    # Step 2: 再畫紅線框（thickness=1）
    mask_u8 = (pred_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (0, 0, 255), thickness=1)

    # 回到 RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

#我的
SIZE = 350
FT_LOW = 0
FT_HIGH = 60
ISOLATION_THRESHOLD = 60

def get_TopK_largest_contours(mask, K=1):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts_sorted[:K]

def fill_holes(mask, K=1):
    TopK_contour = get_TopK_largest_contours(mask, K)
    solid_mask = np.zeros_like(mask)
    if TopK_contour:
        cv2.drawContours(solid_mask, TopK_contour, -1, 255, thickness=cv2.FILLED)
    return solid_mask

def keep_clustered_contours(cnts, sigma=1.5):
    """保留最聚集的輪廓，剔除離群值"""
    if not cnts:
        return []
    
    # 1. 找出每個輪廓的中心點
    centers = []
    valid_cnts_map = [] 
    
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
            valid_cnts_map.append(c)

    if not centers:
        return []

    centers = np.array(centers)
    median_center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - median_center, axis=1)

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + (sigma * std_dist)

    filtered_cnts = []
    for i, dist in enumerate(distances):
        if dist < threshold:
            filtered_cnts.append(valid_cnts_map[i])

    return filtered_cnts

def filter_dark_by_white_center(dark_mask, white_mask, threshold_y=50, threshold_x=100):
    """利用 White Mask (骨頭/脂肪) 的重心來過濾 Dark Mask"""
    M_white = cv2.moments(white_mask)
    if M_white["m00"] == 0:
        return dark_mask 
    
    center_x = int(M_white["m10"] / M_white["m00"])
    center_y = int(M_white["m01"] / M_white["m00"])
    
    cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnts = []
    limit_y = center_y + threshold_y 
    
    for c in cnts:
        M_dark = cv2.moments(c)
        if M_dark["m00"] == 0: continue
        
        cx = int(M_dark["m10"] / M_dark["m00"])
        cy = int(M_dark["m01"] / M_dark["m00"])
        
        if cy > limit_y: continue
        if abs(cx - center_x) > threshold_x: continue
            
        filtered_cnts.append(c)
        
    new_dark_mask = np.zeros_like(dark_mask)
    if filtered_cnts:
        cv2.drawContours(new_dark_mask, filtered_cnts, -1, 255, thickness=cv2.FILLED)
        
    return new_dark_mask

def get_white_mask(img, hand_mask, w_low=150, w_high=255, K=10):
    raw_white_mask = cv2.inRange(img, w_low, w_high)
    white_mask = cv2.bitwise_and(raw_white_mask, raw_white_mask, mask=hand_mask)
    kernel_open = np.ones((3, 3), np.uint8) 
    white_mask = cv2.erode(white_mask, kernel_open, iterations=1)
    # 若 K=1 代表找神經，這裡直接用傳進來的 K
    top_cnts = get_TopK_largest_contours(white_mask, K=K)
    
    final_white_mask = np.zeros_like(raw_white_mask)
    if top_cnts:
        cv2.drawContours(final_white_mask, top_cnts, -1, 255, thickness=cv2.FILLED)
    return final_white_mask

def get_right_half_mask(mask):
    """
    輸入一個遮罩 (例如 CT mask)，回傳該遮罩的「右半部」。
    邏輯：找出 mask 的 Bounding Box，取 x 中點以右的區域。
    """
    # 建立一個全黑遮罩
    right_half_filter = np.zeros_like(mask)
    
    # 找出原始 mask 的輪廓與邊界框 (Bounding Box)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask # 如果原本就是空的，直接回傳
    
    # 合併所有輪廓算出整體的邊界
    all_points = np.vstack(cnts)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # 計算中線 X 座標
    mid_x = x + (w // 2)
    
    # 畫一個白色的矩形，範圍從「中線」到「圖片最右邊」
    img_h, img_w = mask.shape
    cv2.rectangle(right_half_filter, (mid_x, 0), (img_w, img_h), 255, thickness=cv2.FILLED)
    
    # 將「原本的 mask」與「右半部濾鏡」做交集 (Bitwise AND)
    # 這樣就只會剩下 CT mask 右半邊的區域
    final_mask = cv2.bitwise_and(mask, right_half_filter)
    
    return final_mask

def get_dark_mask(img, hand_mask):
    raw_dark_mask = cv2.inRange(img, FT_LOW, FT_HIGH)
    dark_inside_hand = cv2.bitwise_and(raw_dark_mask, raw_dark_mask, mask=hand_mask)
    
    kernel_open = np.ones((3, 3), np.uint8) 
    dark_separated = cv2.erode(dark_inside_hand, kernel_open, iterations=1)
    topk_contours = get_TopK_largest_contours(dark_separated, K=20)
    
    clustered_contours = keep_clustered_contours(topk_contours, sigma=0.1)
    
    final_dark_mask = np.zeros_like(raw_dark_mask)
    if clustered_contours:
        cv2.drawContours(final_dark_mask, clustered_contours, -1, 255, thickness=cv2.FILLED)
    return final_dark_mask

def get_ct_area_mask(dark_mask):
    ct_mask = np.zeros_like(dark_mask)
    cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ct_mask
    
    all_points = np.vstack(cnts)
    hull = cv2.convexHull(all_points)
    cv2.drawContours(ct_mask, [hull], -1, 255, thickness=cv2.FILLED)
    return ct_mask

def generate_masks(img, is_white=False):
    # 影像前處理
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    clache = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    img_clahe = clache.apply(img_blur)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_clahe, -1, kernel)
    
    ret, mask = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_broken = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)
    hand_mask = fill_holes(mask_broken)
    
    if is_white:
        white_mask = get_white_mask(img_sharp, hand_mask)
        return hand_mask, white_mask
    else:
        dark_mask = get_dark_mask(img_sharp, hand_mask)
        return hand_mask, dark_mask

def predict_mask(t1_img, t2_img):
    """
    整合 test.py 的邏輯，一次產生 CT, FT, MN 三種遮罩
    回傳: dict {"CT": mask, "FT": mask, "MN": mask}
    """
    # 1. 前處理 (main.py 呼叫前已做 resize，此處直接用)
    
    # 2. 用 T1 產生 骨頭/脂肪(White) 和 手腕輪廓(ROI)
    mask_roi, mask_white = generate_masks(t1_img, is_white=True)
    
    # 3. 用 T1 產生 原始肌腱(Dark)
    _, mask_dark = generate_masks(t1_img, is_white=False)
    
    # 4. 過濾肌腱雜訊 (位置過濾)
    # 參數：threshold_y=20 (下方容忍度), threshold_x=50 (左右容忍度)
    mask_ft = filter_dark_by_white_center(mask_dark, mask_white, threshold_y=20, threshold_x=50)
    
    # 5. 計算 CT (腕隧道) 區域 (凸包)
    mask_ct = get_ct_area_mask(mask_ft)
    
    # ========================================================
    # 6. 利用 T2 和 CT區域 抓出 正中神經 (MN)
    # ========================================================
    
    # 【修改重點】: 先切出 CT 的「右半部」，縮小搜尋範圍
    mask_ct_right_only = get_right_half_mask(mask_ct)
    
    # 接著只在「右半部的 CT」裡面找最亮的白色 (K=1)
    mask_mn = get_white_mask(t2_img, mask_ct_right_only, w_low=90, w_high=255, K=1)
    
    return {
        "CT": mask_ct,      # 回傳完整的 CT 範圍 (畫圖用完整的比較好看)
        "FT": mask_ft,      # 肌腱
        "MN": mask_mn       # 神經 (只在右半部抓到的)
    }


def dice_coef(gt, pred):
    """
    gt, pred: 0/1 或 bool mask
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    s = gt.sum() + pred.sum()
    if s == 0:
        return 1.0
    return 2.0 * inter / s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation Viewer")
        self.resize(1300, 700)

        # 影像列表
        self.t1_images = []
        self.t2_images = []

        # GT mask（已經 resize + binarize 過的 numpy）
        self.gt_masks = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        # 預測結果 mask（numpy, 0/1）
        self.pred_masks = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        # Dice per image
        self.dice_scores = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        self.idx = 0  # 當前第幾張（0-based）
        self.show_pred = False  # False: 顯示 GT mask; True: 顯示預測結果

        self.setup_ui()

    # ---------------- UI 佈局 ----------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ========== 左邊：T1 / T2 ==========
        left_box = QGroupBox()
        left_layout = QVBoxLayout(left_box)

        left_layout.addWidget(QLabel("T1"))
        self.lbl_t1 = make_image_box()
        left_layout.addWidget(self.lbl_t1)

        left_layout.addWidget(QLabel("T2"))
        self.lbl_t2 = make_image_box()
        left_layout.addWidget(self.lbl_t2)
        left_layout.addStretch()

        # Load T1 + 左右切換
        btn_load_t1 = QPushButton("Load T1 folder")
        btn_prev = QPushButton("←")
        btn_next = QPushButton("→")

        btn_load_t1.clicked.connect(self.load_t1_folder)
        btn_prev.clicked.connect(self.prev_img)
        btn_next.clicked.connect(self.next_img)

        h1 = QHBoxLayout()
        h1.addWidget(btn_load_t1)
        h1.addStretch()
        h1.addWidget(btn_prev)
        h1.addWidget(btn_next)
        left_layout.addLayout(h1)

        # Load T2 + index
        btn_load_t2 = QPushButton("Load T2 folder")
        btn_load_t2.clicked.connect(self.load_t2_folder)

        self.spin_idx = QSpinBox()
        self.spin_idx.setMinimum(0)
        self.spin_idx.setMaximum(0)
        self.spin_idx.setValue(0)
        self.spin_idx.valueChanged.connect(self.go_index)
        
        self.lbl_filename = QLabel("")

        h2 = QHBoxLayout()
        h2.addWidget(btn_load_t2)
        h2.addStretch()
        h2.addWidget(self.spin_idx)
        h2.addWidget(self.lbl_filename)
        left_layout.addLayout(h2)

        # ========== 右邊：Tabs + CT/FT/MN ==========
        right_box = QGroupBox()
        right_layout = QVBoxLayout(right_box)

        self.tabs = QTabWidget()
        self.tab_t1 = QWidget()
        self.tab_t2 = QWidget()
        self.tabs.addTab(self.tab_t1, "T1")
        self.tabs.addTab(self.tab_t2, "T2")
        right_layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # 每個 tab 各有一組 CT/FT/MN 顯示框 + Dice label
        self.result_boxes = {"T1": {}, "T2": {}}
        self.dice_labels = {"T1": {}, "T2": {}}
        self.build_tab("T1", self.tab_t1)
        self.build_tab("T2", self.tab_t2)

        # 下方：Load mask 三顆按鈕 + Predict
        bottom_layout = QHBoxLayout()

        btn_ct_mask = QPushButton("Load CT Mask folder")
        btn_ft_mask = QPushButton("Load FT Mask folder")
        btn_mn_mask = QPushButton("Load MN Mask folder")
        btn_predict = QPushButton("Predict")

        btn_ct_mask.clicked.connect(lambda: self.load_mask_folder("CT"))
        btn_ft_mask.clicked.connect(lambda: self.load_mask_folder("FT"))
        btn_mn_mask.clicked.connect(lambda: self.load_mask_folder("MN"))
        btn_predict.clicked.connect(self.predict_all)

        bottom_layout.addWidget(btn_ct_mask)
        bottom_layout.addWidget(btn_ft_mask)
        bottom_layout.addWidget(btn_mn_mask)
        bottom_layout.addSpacing(40)
        bottom_layout.addWidget(btn_predict)
        bottom_layout.addStretch()

        right_layout.addLayout(bottom_layout)

        # 加到 main layout
        main_layout.addWidget(left_box, 1)
        main_layout.addWidget(right_box, 3)

    # tab 裡面 CT / FT / MN 的三個框
    def build_tab(self, tab_name: str, container: QWidget):
        layout = QVBoxLayout(container)
        grid = QGridLayout()
        grid.setHorizontalSpacing(80)

        titles = ["CT", "FT", "MN"]
        for col, key in enumerate(titles):
            lbl_title = QLabel(key)
            box = make_image_box()
            lbl_dice = QLabel("Dice coefficient:")

            self.result_boxes[tab_name][key] = box
            self.dice_labels[tab_name][key] = lbl_dice

            grid.addWidget(lbl_title, 0, col, alignment=Qt.AlignCenter)
            grid.addWidget(box, 1, col, alignment=Qt.AlignCenter)
            grid.addWidget(lbl_dice, 2, col, alignment=Qt.AlignCenter)

        layout.addLayout(grid)
        layout.addStretch()

    # ---------------- 共用：更新 spin 上限 ----------------
    def update_spin_range(self):
        lengths = [len(self.t1_images), len(self.t2_images)]
        for lst in self.gt_masks.values():
            lengths.append(len(lst))
        max_len = max(lengths) if lengths else 0
        if max_len <= 0:
            self.spin_idx.setMaximum(0)
        else:
            self.spin_idx.setMaximum(max_len - 1)

    # ---------------- 載入影像資料夾 ----------------
    def load_folder_images(self, folder): # 按照檔名排序
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

        # 數字排序
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        return files

    def load_t1_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select T1 Folder")
        if folder:
            self.t1_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    def load_t2_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select T2 Folder")
        if folder:
            self.t2_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    # ---------------- 載入 mask 資料夾 ----------------
    def load_mask_folder(self, kind: str):
        folder = QFileDialog.getExistingDirectory(self, f"Select {kind} Mask Folder")
        if not folder:
            return

        files = self.load_folder_images(folder)
        size = (SIZE, SIZE)
        masks = []

        for path in files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            mask_bin = (img > 127).astype(np.uint8)
            masks.append(mask_bin)

        self.gt_masks[kind] = masks
        # reset 該 kind 的預測
        self.pred_masks[kind] = []
        self.dice_scores[kind] = []

        self.show_pred = False  # 新 mask 進來，先回到 GT 顯示
        self.update_spin_range()
        self.update_base_images()  # 裡面會順便呼叫 update_results()

    # ---------------- 切換 index ----------------
    def prev_img(self):
        if self.idx > 0:
            self.idx -= 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def next_img(self):
        if self.idx < self.spin_idx.maximum():
            self.idx += 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def go_index(self, value):
        self.idx = value
        self.update_base_images()
        
    def update_filename_label(self):
        """
        根據目前 tab + idx 顯示對應影像的檔名
        """
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if base_list and self.idx < len(base_list):
            path = base_list[self.idx]
            name = os.path.basename(path)
            self.lbl_filename.setText(name)
        else:
            self.lbl_filename.setText("")

    # ---------------- 更新左邊 T1/T2 display ----------------
    def update_base_images(self):
        size = SIZE

        # T1
        if self.t1_images and self.idx < len(self.t1_images):
            pix = QPixmap(self.t1_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t1.setPixmap(pix)
        else:
            self.lbl_t1.clear()

        # T2
        if self.t2_images and self.idx < len(self.t2_images):
            pix = QPixmap(self.t2_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t2.setPixmap(pix)
        else:
            self.lbl_t2.clear()

        # 右邊 CT/FT/MN 同步更新
        self.update_results()
        
        # 更新檔案名稱
        self.update_filename_label()

    def on_tab_changed(self, index):
        # 每次切換 T1 / T2，都重新依照目前 tab 更新右側顯示
        self.update_results()
        self.update_filename_label()
    
    # ---------------- 核心：更新 CT / FT / MN 顯示 ----------------
    def update_results(self):
        """
        依照目前 tab (T1 or T2)，將
        - GT mask 或 預測 mask 疊到對應的 T1/T2 影像上
        - 更新 Dice label
        """
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if not base_list or self.idx >= len(base_list):
            for kind in ["CT", "FT", "MN"]:
                self.result_boxes[tab_name][kind].clear()
                self.dice_labels[tab_name][kind].setText("Dice coefficient:")
            return

        base_path = base_list[self.idx]
        base_img = cv2.imread(base_path)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        base_img = cv2.resize(base_img, (SIZE, SIZE))

        for kind in ["CT", "FT", "MN"]:
            box = self.result_boxes[tab_name][kind]
            dice_label = self.dice_labels[tab_name][kind]

            mask_to_use = None
            dice_text = "Dice coefficient:"

            if self.show_pred and self.pred_masks[kind]:
                if self.idx < len(self.pred_masks[kind]):
                    mask_to_use = self.pred_masks[kind][self.idx]
                    if self.idx < len(self.dice_scores[kind]):
                        dice_text = f"Dice coefficient: {self.dice_scores[kind][self.idx]:.3f}"
            
            elif self.gt_masks[kind]:
                if self.idx < len(self.gt_masks[kind]):
                    mask_to_use = self.gt_masks[kind][self.idx]
                    dice_text = "Dice coefficient: -"

            if mask_to_use is None:
                box.clear()
                dice_label.setText("Dice coefficient:")
                continue
            
            # 是否有預測
            if not self.show_pred:
                # 僅顯示 GT mask
                overlay_np = draw_mask(base_img, mask_to_use)

            else:
                # 同時顯示 GT + 預測紅線
                gt = self.gt_masks[kind][self.idx] if self.idx < len(self.gt_masks[kind]) else None
                pred = self.pred_masks[kind][self.idx] if self.idx < len(self.pred_masks[kind]) else None

                if gt is None or pred is None:
                    box.clear()
                    continue
                
                overlay_np = draw_predict_mask(base_img, gt, pred)

            h, w, ch = overlay_np.shape
            bytes_per_line = ch * w
            qimg = QImage(
                overlay_np.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            pix = QPixmap.fromImage(qimg)

            box.setPixmap(pix)
            dice_label.setText(dice_text)

    # ---------------- Predict：針對所有圖做預測 + Dice ----------------
    def predict_all(self):
        """
        針對所有影像執行預測流程，並計算 Dice
        """
        size = (SIZE, SIZE)
        
        # 清空舊的預測結果
        for kind in ["CT", "FT", "MN"]:
            self.pred_masks[kind] = []
            self.dice_scores[kind] = []

        try:
            # 遍歷每一張影像
            # 假設 t1_images 的長度為基準
            count = len(self.t1_images)
            for i in range(count):
                # 1. 讀取並前處理影像
                t1_path = self.t1_images[i]
                t2_path = self.t2_images[i]

                t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
                t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
                
                if t1_img is None or t2_img is None:
                    # 防呆：如果讀不到圖，塞全黑 mask
                    empty = np.zeros(size, dtype=np.uint8)
                    for kind in ["CT", "FT", "MN"]:
                        self.pred_masks[kind].append(empty)
                        self.dice_scores[kind].append(0.0)
                    continue

                t1_img = cv2.resize(t1_img, size)
                t2_img = cv2.resize(t2_img, size)

                # 2. 呼叫整合後的預測函式 (一次取得三個結果)
                # results 是一個字典: {"CT": ..., "FT": ..., "MN": ...}
                results = predict_mask(t1_img, t2_img)

                # 3. 分別儲存結果並計算 Dice
                for kind in ["CT", "FT", "MN"]:
                    pred_bin = results[kind]
                    self.pred_masks[kind].append(pred_bin)
                    
                    # 計算 Dice (如果有載入 GT 的話)
                    if i < len(self.gt_masks[kind]):
                        gt_mask = self.gt_masks[kind][i]
                        d = dice_coef(gt_mask, pred_bin)
                        self.dice_scores[kind].append(d)
                    else:
                        self.dice_scores[kind].append(0.0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "預測錯誤", f"執行預測時發生錯誤：\n{e}")
            return

        self.show_pred = True
        self.update_results()


# ---------------- main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
