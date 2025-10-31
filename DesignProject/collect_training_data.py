# collect_training_data.py
import cv2
import os
import numpy as np

class CameraProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        # 設置較低的解析度以提高效能
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise Exception("無法開啟攝影機")
        print(f"攝影機 {camera_index} 開啟成功")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

class BoardDetector:
    def __init__(self):
        self.min_board_area = 10000  # 降低面積要求

    def detect_board(self, image):
        if image is None:
            return None
            
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自適應二值化
        binary = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # 尋找最大的矩形輪廓
        board_contour = None
        max_area = 0
        
        for contour in contours:
            # 近似輪廓
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 檢查是否是四邊形
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area and area > self.min_board_area:
                    max_area = area
                    board_contour = approx
        
        if board_contour is not None:
            # 透視變換取得正視圖
            warped = self._perspective_transform(image, board_contour)
            return warped
        
        return None

    def _perspective_transform(self, image, contour):
        try:
            # 重新排列輪廓點
            points = contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]
            rect[2] = points[np.argmax(s)]
            
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]
            rect[3] = points[np.argmax(diff)]
            
            # 定義目標點
            width = height = 300
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            # 計算透視變換矩陣
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, matrix, (width, height))
            
            return warped
        except Exception as e:
            print(f"透視變換錯誤: {e}")
            return None

    def _split_cells(self, board_image):
        """分割棋盤為3x3網格"""
        if board_image is None:
            return []
            
        height, width = board_image.shape[:2]
        cell_height = height // 3
        cell_width = width // 3
        
        cells = []
        for i in range(3):
            for j in range(3):
                y_start = i * cell_height + 5  # 減少邊緣緩衝
                y_end = (i + 1) * cell_height - 5
                x_start = j * cell_width + 5
                x_end = (j + 1) * cell_width - 5
                
                cell = board_image[y_start:y_end, x_start:x_end]
                if cell.size > 0:
                    cells.append(cell)
        
        return cells

def create_training_folders():
    """創建訓練資料夾"""
    folders = ['empty', 'red', 'blue']
    for folder in folders:
        os.makedirs(f'training_data/{folder}', exist_ok=True)
    print("訓練資料夾結構已創建")

def save_cells(cells, label, counter):
    """保存格子圖片"""
    saved_count = 0
    for i, cell in enumerate(cells):
        if cell is not None and cell.size > 0:
            filename = f'training_data/{label}/{label}_{counter:04d}_{i}.jpg'
            try:
                cv2.imwrite(filename, cell)
                saved_count += 1
            except Exception as e:
                print(f"保存圖片失敗: {e}")
    
    counter += 1
    print(f"已保存 {saved_count} 張圖片為 {label}")
    return counter

def collect_training_data():
    """主資料收集函數"""
    
    # 創建資料夾
    create_training_folders()
    
    # 嘗試不同的攝影機索引
    camera_index = 0
    camera = None
    
    for i in range(3):  # 嘗試 0, 1, 2
        try:
            camera = CameraProcessor(camera_index=i)
            print(f"成功使用攝影機索引: {i}")
            break
        except Exception as e:
            print(f"攝影機 {i} 無法開啟: {e}")
            continue
    
    if camera is None:
        print("無法開啟任何攝影機，請檢查連接")
        return

    detector = BoardDetector()
    
    print("\n=== 資料收集模式 ===")
    print("按 'e' 保存為空白")
    print("按 'r' 保存為紅色") 
    print("按 'b' 保存為藍色")
    print("按 'c' 顯示/隱藏格子預覽")
    print("按 'q' 退出")
    print("===================\n")
    
    counter = {'empty': 0, 'red': 0, 'blue': 0}
    show_cells = False
    
    try:
        while True:
            # 讀取攝影機畫面
            frame = camera.get_frame()
            if frame is None:
                print("無法讀取攝影機畫面")
                break
            
            # 偵測棋盤
            board_roi = detector.detect_board(frame)
            
            display_frame = frame.copy()
            cells = []
            
            if board_roi is not None:
                # 在畫面上標記棋盤
                cv2.putText(display_frame, "Board Detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 分割格子
                cells = detector._split_cells(board_roi)
                
                if show_cells and cells:
                    # 顯示格子預覽
                    for i, cell in enumerate(cells):
                        if cell is not None and cell.size > 0:
                            row = i // 3
                            col = i % 3
                            # 在主要畫面上顯示小預覽
                            preview_x = 10 + col * 70
                            preview_y = 50 + row * 70
                            
                            # 調整預覽大小
                            preview = cv2.resize(cell, (60, 60))
                            display_frame[preview_y:preview_y+60, preview_x:preview_x+60] = preview
                            
                            # 標記格子編號
                            cv2.putText(display_frame, str(i), (preview_x, preview_y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(display_frame, "No Board Detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 顯示計數器資訊
            info_text = f"Empty: {counter['empty']} | Red: {counter['red']} | Blue: {counter['blue']}"
            cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Collect Data - Press Q to quit', display_frame)
            
            # 鍵盤控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e') and cells:  # 保存為空白
                counter['empty'] = save_cells(cells, 'empty', counter['empty'])
            elif key == ord('r') and cells:  # 保存為紅色
                counter['red'] = save_cells(cells, 'red', counter['red'])
            elif key == ord('b') and cells:  # 保存為藍色
                counter['blue'] = save_cells(cells, 'blue', counter['blue'])
            elif key == ord('c'):  # 切換格子預覽
                show_cells = not show_cells
                print(f"格子預覽: {'開啟' if show_cells else '關閉'}")
    
    except Exception as e:
        print(f"程式執行錯誤: {e}")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"\n資料收集完成！總計:")
        print(f"空白: {counter['empty']} 組")
        print(f"紅色: {counter['red']} 組") 
        print(f"藍色: {counter['blue']} 組")

def simple_collect_mode():
    """簡化版資料收集，不進行棋盤偵測"""
    
    create_training_folders()
    
    try:
        camera = CameraProcessor(0)
    except:
        print("無法開啟攝影機")
        return
    
    print("\n=== 簡化資料收集模式 ===")
    print("直接拍攝整個畫面")
    print("按 'e', 'r', 'b' 保存當前畫面")
    print("按 'q' 退出")
    
    counter = {'empty': 0, 'red': 0, 'blue': 0}
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break
            
            # 顯示簡單的畫面
            display_frame = frame.copy()
            
            # 顯示指引
            cv2.putText(display_frame, "Simple Collection Mode", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press E=Empty, R=Red, B=Blue, Q=Quit", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_text = f"Empty: {counter['empty']} | Red: {counter['red']} | Blue: {counter['blue']}"
            cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Simple Collection Mode', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e'):  # 保存為空白
                filename = f'training_data/empty/empty_{counter["empty"]:04d}.jpg'
                cv2.imwrite(filename, frame)
                counter['empty'] += 1
                print(f"保存空白圖片: {filename}")
            elif key == ord('r'):  # 保存為紅色
                filename = f'training_data/red/red_{counter["red"]:04d}.jpg'
                cv2.imwrite(filename, frame)
                counter['red'] += 1
                print(f"保存紅色圖片: {filename}")
            elif key == ord('b'):  # 保存為藍色
                filename = f'training_data/blue/blue_{counter["blue"]:04d}.jpg'
                cv2.imwrite(filename, frame)
                counter['blue'] += 1
                print(f"保存藍色圖片: {filename}")
    
    except Exception as e:
        print(f"錯誤: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("選擇資料收集模式:")
    print("1. 自動棋盤偵測模式 (推薦)")
    print("2. 簡化模式 (如果自動模式有問題)")
    
    choice = input("請選擇 (1/2): ").strip()
    
    if choice == "1":
        collect_training_data()
    elif choice == "2":
        simple_collect_mode()
    else:
        print("無效選擇，使用自動模式")
        collect_training_data()