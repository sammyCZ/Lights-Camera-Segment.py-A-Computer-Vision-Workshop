import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import tempfile
import yt_dlp

'''
CUSTOMIZE HERE. CHOOSE ANY YOUTUBE VIDEO OR INCLUDE A PATH TO YOUR OWN MP4 FILE
'''
# BG_VIDEO_SOURCE = 'https://www.youtube.com/watch?v=6eRzBNzVohc'
BG_VIDEO_SOURCE = 'backgrounds/video.mp4'
MAX_FAILED_FRAMES = 30


def resolve_background_source(source):
    if not source.startswith(('http://', 'https://')):
        return source
    print("Downloading YouTube clip via yt-dlp...")
    path = os.path.join(tempfile.gettempdir(), "bg_clip.mp4")
    opts = {"quiet": True, "format": "bestvideo[ext=mp4]/bestvideo/best",
            "outtmpl": path, "overwrites": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([source])
    return path


def next_bg_frame(bg_cap, width, height):
    success, frame = bg_cap.read()
    if not success:  # loop the clip back to the start
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = bg_cap.read()
        if not success:
            raise SystemExit("Background video has no readable frames.")
    return cv2.resize(frame, (width, height))


# DirectShow is faster on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the camera. Close other apps using it and try again.")

bg_cap = cv2.VideoCapture(resolve_background_source(BG_VIDEO_SOURCE))
if not bg_cap.isOpened():
    raise SystemExit(f"Could not open background video: {BG_VIDEO_SOURCE}")

failed_frames = 0
with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
    while True:
        success, image = cap.read()
        if not success:
            failed_frames += 1
            print("Ignoring empty camera frame.")
            if failed_frames >= MAX_FAILED_FRAMES:
                raise SystemExit("Camera stopped returning frames. Exiting...")
            continue
        failed_frames = 0

        image = cv2.flip(image, 1)  # selfie view
        # MediaPipe wants RGB; OpenCV uses BGR everywhere else.
        results = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        height, width, _ = image.shape
        condition = results.segmentation_mask[..., None] > 0.1  # (h, w, 1) broadcasts over RGB
        bg_image = next_bg_frame(bg_cap, width, height)
        output_image = np.where(condition, image, bg_image)

        cv2.imshow("Lights, Camera, Segment!", output_image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
bg_cap.release()
cv2.destroyAllWindows()
