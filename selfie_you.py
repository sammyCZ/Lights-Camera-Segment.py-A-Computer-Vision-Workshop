# Create and enter virtual environment first
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import tempfile
import yt_dlp

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Path to the "movie scene" clip you want to composite onto.
# Can be a local file path OR a YouTube link.
BG_VIDEO_SOURCE = "https://www.youtube.com/watch?v=ODmhPsgqGgQ"


def resolve_background_source(source):
    if not source.startswith(("http://", "https://")):
        return source
    print("Downloading YouTube clip via yt-dlp...")
    tmp_path = os.path.join(tempfile.gettempdir(), "selfie2_bg_clip.mp4")
    ydl_opts = {
        "quiet": True,
        "format": "bestvideo[ext=mp4]/bestvideo/best",
        "outtmpl": tmp_path,
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([source])
    return tmp_path


# added for support. DirectShow works faster
if sys.platform == "win32":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit(
        "Could not open the camera. Close other apps using it and try again."
    )

bg_cap = cv2.VideoCapture(resolve_background_source(BG_VIDEO_SOURCE))
if not bg_cap.isOpened():
    raise SystemExit(f"Could not open background video: {BG_VIDEO_SOURCE}")

def next_bg_frame(width, height):
    success, frame = bg_cap.read()
    if not success:
        # loop the clip back to the start
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = bg_cap.read()
        if not success:
            raise SystemExit("Background video has no readable frames.")
    return cv2.resize(frame, (width, height))

MAX_FAILED_FRAMES = 30
failed_frames = 0
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            failed_frames += 1
            print("Ignoring empty camera frame.")
            if failed_frames >= MAX_FAILED_FRAMES:
                raise SystemExit("Camera stopped returning frames. Exiting...")
            continue
        failed_frames = 0

        # Flip for a selfie-view, convert BGR -> RGB for MediaPipe.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        bg_image = next_bg_frame(image_width, image_height)
        output_image = np.where(condition, image, bg_image)

        cv2.imshow("Lights, Camera, Segment!", output_image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
bg_cap.release()
cv2.destroyAllWindows()
