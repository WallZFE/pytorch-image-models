import cv2
import numpy as np
import sys

# ============== 配置参数 ==============
BEV_WIDTH = 600    # 鸟瞰图输出宽度 (像素)
BEV_HEIGHT = 800   # 鸟瞰图输出高度 (像素)
# ======================================

def main(image_path, points):
    global image, image_clone
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        sys.exit(1)
 
    image_clone = image.copy()
    h, w = image.shape[:2]
    print(f"图片尺寸: {w} x {h}")
    
    # ---- 源点 (原图上的4个点) ----
    src_pts = np.float32(points)
    
    # ---- 目标点 (鸟瞰图上的4个点，构成矩形 -> 车道线平行) ----
    # 对应顺序: 左下, 右下, 左上, 右上
    dst_pts = np.float32([
        [0, BEV_HEIGHT],                # 左近 -> 左下
        [BEV_WIDTH, BEV_HEIGHT],        # 右近 -> 右下
        [0, 0],                         # 左远 -> 左上
        [BEV_WIDTH, 0]                  # 右远 -> 右上
    ])
    
    # ---- 计算透视变换矩阵 (3x3) ----
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    print("\n变换矩阵 H (3x3):")
    print(H)
    
    # ---- 对原图做透视变换，生成鸟瞰图 ----
    bev_image = cv2.warpPerspective(image, H, (BEV_WIDTH, BEV_HEIGHT))
    
    # ---- 在原图上画车道线区域 ----
    overlay = image.copy()
    pts_poly = np.array([points[0], points[1], points[3], points[2]], dtype=np.int32)
    cv2.fillPoly(overlay, [pts_poly], (0, 255, 0, 80))
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
    
    # 画点和连线
    for i, pt in enumerate(points):
        cv2.circle(image, tuple(map(int, pt)), 8, (0, 0, 255), -1)
        cv2.putText(image, str(i + 1), (int(pt[0]) + 10, int(pt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.polylines(image, [np.array(points[:2], dtype=np.int32)], False, (255, 0, 0), 2)
    cv2.polylines(image, [np.array(points[2:], dtype=np.int32)], False, (255, 0, 0), 2)
    cv2.line(image, tuple(map(int, points[0])), tuple(map(int, points[2])), (255, 0, 0), 2)
    cv2.line(image, tuple(map(int, points[1])), tuple(map(int, points[3])), (255, 0, 0), 2)
    
    # ---- 拼接原图和鸟瞰图 ----
    # 将原图缩放到和BEV同高
    scale = BEV_HEIGHT / h
    resized_orig = cv2.resize(image, (int(w * scale), BEV_HEIGHT))
    
    # 在鸟瞰图上画辅助线
    cv2.line(bev_image, (BEV_WIDTH // 2, 0), (BEV_WIDTH // 2, BEV_HEIGHT), (0, 255, 255), 1)
    cv2.rectangle(bev_image, (0, 0), (BEV_WIDTH - 1, BEV_HEIGHT - 1), (255, 255, 255), 2)
    
    combined = np.hstack([resized_orig, bev_image])
    
    # ---- 保存结果 ----
    cv2.imwrite("bev_result.jpg", combined)
    print(f"\n✅ 效果图已保存: bev_result.jpg")
    
    # 保存变换矩阵到文本文件 (C语言可读格式)
    with open("homography_matrix.txt", "w") as f:
        f.write(f"# Bird's Eye View Homography Matrix (3x3)\n")
        f.write(f"# BEV Output Size: {BEV_WIDTH} x {BEV_HEIGHT}\n")
        f.write(f"# Source points (original image):\n")
        for i, pt in enumerate(points):
            f.write(f"#   Point {i+1}: ({pt[0]}, {pt[1]})\n")
        f.write(f"#\n")
        f.write(f"# Matrix values (row-major, 9 floats):\n")
        for i in range(3):
            row_vals = " ".join([f"{H[i, j]:.10f}" for j in range(3)])
            f.write(f"{row_vals}\n")
    
    print(f"✅ 变换矩阵已保存: homography_matrix.txt")
    print(f"\n鸟瞰图尺寸: {BEV_WIDTH} x {BEV_HEIGHT}")
    print("请检查 bev_result.jpg 中右侧的鸟瞰图，车道线是否平行且垂直。")

if __name__ == "__main__":
    image_path = "road.jpg"

    '''
    规则: 选择最外侧两条车道线 把四条车道线都包含进去
        远处y坐标点要等于35% 这样后面判断变换后的y坐标小于0的都可以丢弃
    print("  1. 左车道线 近端 (底部左侧)")
    print("  2. 右车道线 近端 (底部右侧)")
    print("  3. 左车道线 远端 (上方左侧)")
    print("  4. 右车道线 远端 (上方右侧)")
    '''
    coords = [[1.85, 244.17], [609.02, 244.18], [282.13, 126.02], [316.40, 126.00]]
    main(image_path, coords)