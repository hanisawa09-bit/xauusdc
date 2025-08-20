import os
import sys
import time
import subprocess
import numpy as np
import cv2
import requests
from datetime import datetime

# ======================= USER SETTINGS ==========================
YOUTUBE_URL = "https://www.youtube.com/watch?v=851TJ5-Syz4"

# Telegram
BOT_TOKEN = "8357139228:AAFYwfrHAsqm7GnJ6g1qmtfNN9DwIivNIiE"
CHAT_ID = "7669149607"

# Processing / capture
TARGET_FPS = 1
FORCED_W, FORCED_H = 1280, 720
TOP_PCT, BTM_PCT = 0.12, 0.12
LFT_PCT, RGT_PCT = 0.12, 0.055
MIN_COLOR_PIXELS = 50
SEG_THRESHOLD_RATIO = 0.35
MIN_SEG_WIDTH = 3
APPROX_CANDLES_ON_SCREEN = 100

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)
# ================================================================

# ---------------------- Telegram helpers ------------------------
def send_telegram_message(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=10,
        )
    except Exception as e:
        print("Telegram message error:", e, file=sys.stderr)

def send_telegram_image(filename: str, caption: str = ""):
    try:
        with open(filename, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": f},
                timeout=20,
            )
    except Exception as e:
        print("Telegram image error:", e, file=sys.stderr)

# ---------------------- Stream resolving ------------------------
def get_stream_url_via_streamlink(url: str) -> str:
    import streamlink
    session = streamlink.Streamlink()
    session.set_option("http-headers", "User-Agent=Mozilla/5.0")
    streams = session.streams(url)
    if not streams:
        raise RuntimeError("No playable streams found via Streamlink.")
    if "best" in streams:
        return streams["best"].url
    qualities = sorted(streams.items(), key=lambda kv: kv[0])
    return qualities[-1][1].url

# ------------------- Fallback ffmpeg (MJPEG) --------------------
def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    from shutil import which
    ff = which("ffmpeg")
    if ff:
        return ff
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError("FFmpeg not found.")

def capture_one_frame(stream_url: str):
    ffmpeg = get_ffmpeg_path()
    vf = f"fps=1,scale={FORCED_W}:{FORCED_H}:flags=bicubic"
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "30",
        "-i", stream_url,
        "-vframes", "1",
        "-vf", vf,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate(timeout=20)
    if not out:
        return None
    arr = np.frombuffer(out, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

# ----------------------- Candle detection -----------------------
def robust_second_last_candle_roi(chart_img: np.ndarray):
    h, w = chart_img.shape[:2]
    hsv = cv2.cvtColor(chart_img, cv2.COLOR_BGR2HSV)
    colored = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))
    col_counts = np.count_nonzero(colored, axis=0)
    k = max(5, w // 200)
    kernel = np.ones(k, dtype=np.float32) / k
    smoothed = np.convolve(col_counts.astype(np.float32), kernel, mode="same")
    mx = float(smoothed.max()) if smoothed.size else 0.0
    if mx <= 0:
        cw = max(1, w // max(5, APPROX_CANDLES_ON_SCREEN))
        x2 = w - 2 * cw
        return max(0, x2), min(w, x2 + cw)
    thr = max(10.0, SEG_THRESHOLD_RATIO * mx)
    mask = smoothed > thr
    segments = []
    i = 0
    while i < w:
        if mask[i]:
            j = i + 1
            while j < w and mask[j]:
                j += 1
            if (j - i) >= MIN_SEG_WIDTH:
                segments.append((i, j))
            i = j
        else:
            i += 1
    if len(segments) >= 2:
        left, right = segments[-2]
        pad = max(1, (right - left) // 8)
        return max(0, left - pad), min(w, right + pad)
    cw = max(1, w // max(5, APPROX_CANDLES_ON_SCREEN))
    x2 = w - 2 * cw
    return max(0, x2), min(w, x2 + cw)

# ----------------------- Candle Color ---------------------------
def get_candle_color(candle_img: np.ndarray) -> str:
    hsv = cv2.cvtColor(candle_img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
    red = cv2.add(red1, red2)
    green = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))
    pink = cv2.inRange(hsv, (130, 50, 50), (160, 255, 255))

    kernel = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, kernel, iterations=1)

    # count pixels
    red_count = cv2.countNonZero(red)
    green_count = cv2.countNonZero(green)
    pink_count = cv2.countNonZero(pink)

    # if pink present with other colors, pink dominates
    if pink_count >= MIN_COLOR_PIXELS:
        return "Pink"
    elif red_count >= MIN_COLOR_PIXELS and red_count > green_count:
        return "Red"
    elif green_count >= MIN_COLOR_PIXELS:
        return "Green"
    else:
        return "Unknown"

def color_to_signal(color: str) -> str:
    return {"Red": "🔴 SELL", "Green": "🟢 BUY", "Pink": "⚪ EXIT"}.get(color, "Unknown")

def draw_and_save_snapshot(chart_img: np.ndarray, x1: int, x2: int, detected_color: str) -> str:
    dbg = chart_img.copy()
    h, w = dbg.shape[:2]
    cv2.rectangle(dbg, (x1, 0), (x2, h), (0, 255, 255), 3)
    label = f"2nd-from-right: {detected_color}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    cv2.rectangle(dbg, (10, 10), (10 + 600, 50), (0, 0, 0), -1)
    cv2.putText(dbg, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    fname = os.path.join(SNAP_DIR, f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(fname, dbg)
    return fname

# ----------------------------- Main -----------------------------
def main():
    send_telegram_message("🟢 XAUUSDc Bot starting: YouTube live check every 1 min")
    last_color = None

    while True:
        try:
            play_url = get_stream_url_via_streamlink(YOUTUBE_URL)
            print("Resolved stream URL:", play_url)

            frame = capture_one_frame(play_url)
            if frame is None:
                raise RuntimeError("Failed to capture frame.")

            h, w = frame.shape[:2]
            top = int(TOP_PCT * h)
            btm = int(BTM_PCT * h)
            lft = int(LFT_PCT * w)
            rgt = int(RGT_PCT * w)
            chart = frame[top:h - btm, lft:w - rgt]
            if chart.size == 0:
                chart = frame

            x1, x2 = robust_second_last_candle_roi(chart)
            candle_img = chart[:, x1:x2]
            color = get_candle_color(candle_img)

            if color != "Unknown":
                if last_color is None:
                    snap = draw_and_save_snapshot(chart, x1, x2, color)
                    send_telegram_image(snap, caption=f"🔔 XAUUSDc Started | Candle color: {color} | {color_to_signal(color)}")
                    last_color = color
                elif color != last_color:
                    snap = draw_and_save_snapshot(chart, x1, x2, color)
                    send_telegram_image(snap, caption=f"🔔 XAUUSDc Candle changed {last_color} → {color} | {color_to_signal(color)}")
                    last_color = color

            # wait 1 minute before next check
            time.sleep(60)

        except KeyboardInterrupt:
            print("Exiting…")
            break
        except Exception as e:
            print("Error:", e, " — retrying in 10s…", file=sys.stderr)
            send_telegram_message(f"⚠️ Error: {e} — retrying…")
            time.sleep(10)
            continue

if __name__ == "__main__":
    main()
