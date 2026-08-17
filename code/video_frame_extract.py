import cv2
import os
import re
import glob
import subprocess
import shutil
from pathlib import Path
import numpy as np

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("⚠ decord 未安装，H.265 将回退到 FFmpeg 解码。pip install decord")

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


# ──────────────────────────────────────────────
#  文件发现
# ──────────────────────────────────────────────

def get_video_files(input_dir, video_extensions=None):
    """获取目录下所有普通视频文件（含 H.265 容器格式）"""
    if video_extensions is None:
        video_extensions = [
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv',
            '.mpeg', '.mpg', '.m4v', '.ts', '.m2ts', '.webm',
        ]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        video_files.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))
    return sorted(set(video_files))


def get_yuv_files(input_dir, yuv_extensions=None):
    """获取目录下所有 YUV 原始视频文件"""
    if yuv_extensions is None:
        yuv_extensions = ['.yuv', '.nv21', '.nv12', '.raw']
    yuv_files = []
    for ext in yuv_extensions:
        yuv_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        yuv_files.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))
    return sorted(set(yuv_files))


def get_h265_files(input_dir, h265_extensions=None):
    """
    获取目录下所有 H.265/HEVC 裸流文件
    常见扩展名: .h265, .hevc, .265, .h265_raw
    """
    if h265_extensions is None:
        h265_extensions = ['.h265', '.hevc', '.265']
    h265_files = []
    for ext in h265_extensions:
        h265_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        h265_files.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))
    return sorted(set(h265_files))


# ──────────────────────────────────────────────
#  前缀提取（复用原逻辑）
# ──────────────────────────────────────────────

def _get_prefix(file_path):
    """从文件路径中提取时间戳前缀或清理后的文件名前缀"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}h\d{2}m\d{2}s-)', file_path)
    if match:
        return match.group(1)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    return re.sub(r'[^\w\-_]', '_', stem) + '_'


# ──────────────────────────────────────────────
#  普通视频抽帧（OpenCV，原逻辑）
# ──────────────────────────────────────────────

def extract_frames_from_video(video_path, output_folder, frame_interval=1):
    """从单个视频中按间隔抽帧（OpenCV 后端）"""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠ OpenCV 无法打开，尝试 decord ...")
        return extract_frames_from_video_decord(video_path, output_folder, frame_interval)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  视频信息: {os.path.basename(video_path)}")
    print(f"  分辨率: {width}x{height}, FPS: {fps:.2f}, 总帧数: {total_frames}")

    prefix = _get_prefix(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_folder, f"{prefix}frame_{frame_count:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

    cap.release()
    print(f"  完成! 共保存 {saved_count} 帧")
    return saved_count


# ──────────────────────────────────────────────
#  H.265 解码 —— 方案 A: decord
# ──────────────────────────────────────────────

def extract_frames_from_video_decord(video_path, output_folder, frame_interval=1):
    """
    使用 decord 解码视频（天然支持 H.265/HEVC）并抽帧。
    适用于: .mp4/.mkv/.ts 等容器中的 HEVC 编码，以及部分裸流。
    """
    if not HAS_DECORD:
        print("  ⚠ decord 不可用，跳过")
        return 0

    os.makedirs(output_folder, exist_ok=True)

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"  ⚠ decord 打开失败: {e}")
        return 0

    total_frames = len(vr)
    fps = vr.get_avg_fps()
    h, w = vr[0].shape[:2]

    print(f"  [decord] 视频信息: {os.path.basename(video_path)}")
    print(f"  分辨率: {w}x{h}, FPS: {fps:.2f}, 总帧数: {total_frames}")

    prefix = _get_prefix(video_path)
    saved_count = 0

    # 构造需要读取的帧索引列表，避免一次性全部加载
    indices = list(range(0, total_frames, frame_interval))

    # 分批读取，防止内存溢出
    batch_size = 256
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        frames = vr.get_batch(batch_idx).asnumpy()  # (N, H, W, 3) RGB

        for j, idx in enumerate(batch_idx):
            bgr = cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR)
            out_path = os.path.join(output_folder, f"{prefix}frame_{idx:06d}.jpg")
            cv2.imwrite(out_path, bgr)
            saved_count += 1

        done = start + len(batch_idx)
        print(f"  进度: {done}/{total_frames} ({done/total_frames*100:.1f}%)")

    print(f"  完成! 共保存 {saved_count} 帧")
    return saved_count


# ──────────────────────────────────────────────
#  H.265 解码 —— 方案 B: FFmpeg 子进程
# ──────────────────────────────────────────────

def _find_ffmpeg():
    """查找 ffmpeg 可执行文件路径"""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        # 常见 Windows 安装路径
        for p in [r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
            if os.path.isfile(p):
                return p
    return ffmpeg_path


def extract_frames_from_h265_ffmpeg(h265_path, output_folder, frame_interval=1,
                                     width=None, height=None, fps=None):
    """
    使用 FFmpeg 解码 H.265 裸流 / 容器视频并抽帧。
    
    参数:
        h265_path:      H.265 文件路径（裸流 .h265/.hevc 或容器 .mp4/.ts 等）
        output_folder:  输出目录
        frame_interval: 帧间隔
        width/height:   裸流时必须指定分辨率（容器格式可自动检测）
        fps:            裸流时建议指定帧率（默认25）
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        print("  ⚠ 未找到 ffmpeg，无法解码 H.265 裸流")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    prefix = _get_prefix(h265_path)

    # ---------- 构造 ffmpeg 命令 ----------
    cmd = [ffmpeg, "-y"]

    file_ext = os.path.splitext(h265_path)[1].lower()
    is_raw = file_ext in ('.h265', '.hevc', '.265')

    if is_raw:
        # 裸 H.265 流：需要手动指定输入格式
        cmd += ["-f", "hevc"]
        if fps:
            cmd += ["-r", str(fps)]
        cmd += ["-i", h265_path]
    else:
        # 容器格式，ffmpeg 自动识别
        cmd += ["-i", h265_path]

    # 抽帧滤镜：每 frame_interval 帧取 1 帧
    if frame_interval > 1:
        cmd += ["-vf", f"select='not(mod(n\\,{frame_interval}))'", "-vsync", "vfr"]

    # 输出 JPEG 序列
    out_pattern = os.path.join(output_folder, f"{prefix}frame_%06d.jpg")
    cmd += ["-q:v", "2", out_pattern]

    print(f"  [FFmpeg] 解码 H.265: {os.path.basename(h265_path)}")
    print(f"  命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600,   # 1 小时超时
        )
        if result.returncode != 0:
            stderr_text = result.stderr.decode(errors='replace')
            print(f"  ⚠ FFmpeg 报错:\n{stderr_text[-500:]}")
            return 0
    except subprocess.TimeoutExpired:
        print("  ⚠ FFmpeg 超时")
        return 0
    except Exception as e:
        print(f"  ⚠ FFmpeg 执行异常: {e}")
        return 0

    # 统计保存的帧数
    saved = len(glob.glob(os.path.join(output_folder, f"{prefix}frame_*.jpg")))
    print(f"  完成! 共保存 {saved} 帧")
    return saved


# ──────────────────────────────────────────────
#  H.265 解码 —— 方案 C: PyAV（可选备选）
# ──────────────────────────────────────────────

def extract_frames_from_video_pyav(video_path, output_folder, frame_interval=1):
    """使用 PyAV 解码视频并抽帧（支持 H.265）"""
    if not HAS_PYAV:
        print("  ⚠ PyAV 不可用")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    prefix = _get_prefix(video_path)

    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"  ⚠ PyAV 打开失败: {e}")
        return 0

    stream = container.streams.video[0]
    total_frames = stream.frames or 0
    fps = float(stream.average_rate or 25)
    codec_name = stream.codec_context.name

    print(f"  [PyAV] 视频信息: {os.path.basename(video_path)}")
    print(f"  编码: {codec_name}, FPS: {fps:.2f}, 总帧数: {total_frames or '未知'}")

    frame_count = 0
    saved_count = 0

    for frame in container.decode(video=0):
        if frame_count % frame_interval == 0:
            img = frame.to_ndarray(format='bgr24')
            out_path = os.path.join(output_folder, f"{prefix}frame_{frame_count:06d}.jpg")
            cv2.imwrite(out_path, img)
            saved_count += 1
        frame_count += 1
        if frame_count % 100 == 0:
            info = f"{frame_count}/{total_frames}" if total_frames else f"{frame_count}"
            print(f"  进度: {info} 帧")

    container.close()
    print(f"  完成! 共保存 {saved_count} 帧")
    return saved_count


# ──────────────────────────────────────────────
#  H.265 统一入口：自动选择最佳解码器
# ──────────────────────────────────────────────

def extract_frames_from_h265(h265_path, output_folder, frame_interval=1,
                              width=None, height=None, fps=None):
    """
    H.265 视频统一抽帧入口。
    自动按优先级尝试: decord → PyAV → FFmpeg → OpenCV
    
    参数:
        h265_path:      H.265 视频文件路径
        output_folder:  输出目录
        frame_interval: 帧间隔
        width/height:   裸流分辨率（可选）
        fps:            裸流帧率（可选）
    """
    file_ext = os.path.splitext(h265_path)[1].lower()
    is_raw = file_ext in ('.h265', '.hevc', '.265')

    print(f"\n  ▶ 处理 H.265 文件: {os.path.basename(h265_path)}")
    print(f"    类型: {'裸 H.265 流' if is_raw else 'H.265 容器格式'}")

    # 裸流 → 优先 FFmpeg（最稳定）
    if is_raw:
        saved = extract_frames_from_h265_ffmpeg(
            h265_path, output_folder, frame_interval, width, height, fps
        )
        if saved > 0:
            return saved
        print("  ⚠ FFmpeg 失败，尝试其他方式...")

    # 容器格式 / 裸流回退 → decord
    if HAS_DECORD:
        saved = extract_frames_from_video_decord(h265_path, output_folder, frame_interval)
        if saved > 0:
            return saved

    # → PyAV
    if HAS_PYAV:
        saved = extract_frames_from_video_pyav(h265_path, output_folder, frame_interval)
        if saved > 0:
            return saved

    # → FFmpeg（容器格式也走一遍）
    if not is_raw:
        saved = extract_frames_from_h265_ffmpeg(
            h265_path, output_folder, frame_interval, width, height, fps
        )
        if saved > 0:
            return saved

    # → OpenCV 兜底
    saved = extract_frames_from_video(h265_path, output_folder, frame_interval)
    return saved


# ──────────────────────────────────────────────
#  YUV 抽帧（原逻辑不变）
# ──────────────────────────────────────────────

def extract_frames_from_yuv(yuv_path, output_folder, width=3840, height=2160,
                            yuv_format='nv21', frame_interval=1):
    os.makedirs(output_folder, exist_ok=True)
    yuv_format = yuv_format.lower()
    if yuv_format not in ('nv21', 'nv12', 'i420', 'yv12'):
        print(f"不支持的YUV格式: {yuv_format}")
        return 0

    frame_size = width * height * 3 // 2
    file_size = os.path.getsize(yuv_path)
    total_frames = file_size // frame_size
    remainder = file_size % frame_size

    print(f"  YUV: {os.path.basename(yuv_path)}, {width}x{height}, {yuv_format.upper()}")
    print(f"  总帧数: {total_frames}")
    if remainder:
        print(f"  ⚠ 末尾 {remainder} 字节不足一帧，已忽略")
    if total_frames == 0:
        return 0

    prefix = _get_prefix(yuv_path)
    saved_count = 0

    with open(yuv_path, 'rb') as f:
        for idx in range(total_frames):
            raw = f.read(frame_size)
            if len(raw) < frame_size:
                break
            yuv_arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = _convert_yuv_to_bgr(yuv_arr, width, height, yuv_format)

            if idx % frame_interval == 0:
                out = os.path.join(output_folder, f"{prefix}frame_{idx:06d}.jpg")
                cv2.imwrite(out, bgr)
                saved_count += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == total_frames:
                print(f"  进度: {idx+1}/{total_frames} ({(idx+1)/total_frames*100:.1f}%)")

    print(f"  完成! 共保存 {saved_count} 帧")
    return saved_count


def _convert_yuv_to_bgr(yuv_array, width, height, yuv_format):
    if yuv_format == 'nv21':
        yuv = yuv_array.reshape((height + height // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    elif yuv_format == 'nv12':
        yuv = yuv_array.reshape((height + height // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif yuv_format == 'i420':
        yuv = yuv_array.reshape((height + height // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    elif yuv_format == 'yv12':
        y_size = width * height
        uv_size = y_size // 4
        y_p = yuv_array[:y_size]
        v_p = yuv_array[y_size:y_size + uv_size]
        u_p = yuv_array[y_size + uv_size:]
        i420 = np.concatenate([y_p, u_p, v_p])
        yuv = i420.reshape((height + height // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    else:
        raise ValueError(f"不支持的YUV格式: {yuv_format}")


# ──────────────────────────────────────────────
#  批量处理入口（整合所有格式）
# ──────────────────────────────────────────────

def batch_extract_frames(input_dir, output_base_dir, frame_interval=15,
                         h265_width=None, h265_height=None, h265_fps=None):
    """
    批量处理目录下的所有视频文件
    
    参数:
        input_dir:       输入目录
        output_base_dir: 输出基础目录
        frame_interval:  帧间隔
        h265_width:      H.265 裸流宽度（可选）
        h265_height:     H.265 裸流高度（可选）
        h265_fps:        H.265 裸流帧率（可选，默认25）
    """
    video_files = get_video_files(input_dir)
    yuv_files   = get_yuv_files(input_dir)
    h265_files  = get_h265_files(input_dir)

    all_files = video_files + h265_files + yuv_files  # H.265 裸流单独分类

    if not all_files:
        print(f"在目录 {input_dir} 中没有找到任何视频文件")
        return

    print(f"找到 {len(video_files)} 个普通视频 + {len(h265_files)} 个H.265裸流 + {len(yuv_files)} 个YUV文件:")
    idx = 1
    for f in video_files:
        print(f"  {idx}. [视频]  {os.path.basename(f)}"); idx += 1
    for f in h265_files:
        print(f"  {idx}. [H265]  {os.path.basename(f)}"); idx += 1
    for f in yuv_files:
        print(f"  {idx}. [YUV]   {os.path.basename(f)}"); idx += 1

    os.makedirs(output_base_dir, exist_ok=True)

    total = len(all_files)
    total_saved = 0

    print(f"\n{'='*60}")
    print(f"开始批量处理")
    print(f"输入: {input_dir}")
    print(f"输出: {output_base_dir}")
    print(f"帧间隔: {frame_interval}")
    print(f"{'='*60}\n")

    h265_exts = {'.h265', '.hevc', '.265'}
    yuv_exts  = {'.yuv', '.nv21', '.nv12', '.raw'}

    for i, fpath in enumerate(all_files, 1):
        fname = os.path.basename(fpath)
        fext  = os.path.splitext(fname)[1].lower()

        print(f"\n[{i}/{total}] {fname}")

        clean_name = re.sub(r'[^\w\-_\.]', '_', Path(fname).stem)
        out_folder = os.path.join(output_base_dir, clean_name)

        # ── H.265 裸流 ──
        if fext in h265_exts:
            saved = extract_frames_from_h265(
                fpath, out_folder, frame_interval,
                width=h265_width, height=h265_height, fps=h265_fps,
            )

        # ── YUV 原始数据 ──
        elif fext in yuv_exts:
            saved = extract_frames_from_yuv(
                fpath, out_folder,
                width=3840, height=2160,
                yuv_format='nv21', frame_interval=frame_interval,
            )

        # ── 普通视频 / H.265 容器 ──
        else:
            # 先检测编码是否为 HEVC，若是则走 H.265 专用流程
            codec = _probe_codec(fpath)
            if codec and codec.lower() in ('hevc', 'h265', 'hev1', 'hvc1'):
                print(f"  检测到 HEVC 编码，使用 H.265 解码流程")
                saved = extract_frames_from_h265(fpath, out_folder, frame_interval)
            else:
                saved = extract_frames_from_video(fpath, out_folder, frame_interval)

        total_saved += saved

    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"总文件数: {total}")
    print(f"总保存帧数: {total_saved}")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*60}")


def _probe_codec(video_path):
    """
    用 ffprobe 探测视频编码格式，返回编码名称字符串（如 'hevc'）。
    若 ffprobe 不可用则返回 None。
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "csv=p=0",
             video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        return result.stdout.decode().strip() or None
    except Exception:
        return None


# ──────────────────────────────────────────────
#  主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    input_directory  = "../../data/dispose/"
    output_base_directory = "../../data/dispose/result/"
    frame_interval   = 1

    batch_extract_frames(
        input_dir=input_directory,
        output_base_dir=output_base_directory,
        frame_interval=frame_interval,
        h265_width=3840,
        h265_height=2160,
        h265_fps=25,
    )