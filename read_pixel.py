import cv2
import numpy as np

def check_images_identical(image_path, rows, cols):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片，请检查路径是否正确。")
        return

    height, width, channels = img.shape
    print(f"原图尺寸: 宽={width}, 高={height}")
    
    # 2. 计算每个子图的宽和高
    # 假设各子图大小严格一致且无缝拼接
    h_step = 224
    w_step = 224
    
    if cols < 4:
        print("错误：图片的列数少于 4，不存在第四张图。")
        return

    # 3. 截取下排（最后一行）的第一张图和第四张图
    # 下排的 y 坐标范围
    y_start = (rows - 1) * h_step
    y_end = y_start + h_step
    
    # 第一张图的 x 坐标范围 (索引 0)
    img1 = img[y_start+2:y_end+2, 2:w_step+2]
    
    # 第四张图的 x 坐标范围 (索引 3)
    img4 = img[y_start+2:y_end+2, 3*w_step+2 : 4*w_step+2]
    
    # 4. 判断像素是否完全相同
    if img1.shape != img4.shape:
        print("截取出的两张图片尺寸不一致，不相同。")
        return
        
    # np.array_equal 会严格逐个像素进行对比
    if np.array_equal(img1, img4):
        print("✅ 结论: 下排的第一张图和第四张图的像素【完全相同】。")
    else:
        print("❌ 结论: 下排的第一张图和第四张图的像素【不完全相同】。")
        
        # 可选：计算差异程度
        diff = cv2.absdiff(img1, img4)
        non_zero_count = np.count_nonzero(diff)
        total_pixels = img1.size
        print(f"不同像素值数量占总数据的: {non_zero_count}/{total_pixels}")

# --- 使用示例 ---
# 假设这张图被分成了 2 行 5 列，你需要根据你实际图片的网格数来修改 rows 和 cols
image_path = "debug_output/step_8500_stage_1_jitter.png"
check_images_identical(image_path, rows=2, cols=16)