# Lights, Camera, Segment.py: A Computer Vision Workshop

A hands-on workshop teaching the basics of computer vision using background
segmentation as the running example. We walk through three segmentation
approaches, in order of increasing complexity:

1. **rembg**: background removal on a single static image
2. **MediaPipe Selfie Segmentation**: the same idea, but applied live to a
   webcam video feed, to show that these models aren't limited to still images
3. **selfie2.py**: segmentation applied live to a webcam feed, composited
   onto a moving *video* background (a local clip or a YouTube link) instead
   of a solid color, so it looks like you're standing inside the scene

## What this covers

- Installing Python and setting up an isolated environment for CV work
- Removing the background from a static image with `rembg`
- Choosing a CPU vs. GPU backend for a model, and why that choice matters
- Running a pretrained segmentation model (MediaPipe Selfie Segmentation) on
  live webcam input
- Swapping/customizing the background behind a person in real time
- Compositing a live person onto a moving background video by pulling that background straight from a YouTube link

## Overview of how this repo works

All three scripts are the same basic recipe, just applied to different inputs:

1. A model looks at an image/frame and returns a **mask** — for every pixel,
   roughly "how likely is this part of the person?" (`segmentation_mask`).
2. We turn that into a yes/no array by thresholding it (`> 0.1`), called
   `condition` in `selfie.py`/`selfie2.py`.
3. `np.where(condition, A, B)` builds the final image pixel-by-pixel: wherever
   `condition` is true, take the pixel from `A` (the person); otherwise take
   it from `B` (the new background).

For this to work, `condition`, `A`, and `B` all need the exact same shape —
that's why `output_image = (1080, 1920, 3)` failing against a `(6667, 10000,
3)` background earlier was a crash: `np.where` has no idea how to line up two
arrays of different sizes, so anything you composite in has to be resized to
match the webcam frame first.

`segment.py`'s comments walk through the same idea for a still image, plus
why the output needs a 4th (alpha/transparency) channel and why that only
saves cleanly as `.png`.

`selfie.py` and `selfie2.py` build `condition` from the mask in two slightly
different, equally valid ways — worth a look side by side if you want to see
the same idea written two ways:

- `selfie.py`: `np.stack((mask,) * 3, axis=-1) > 0.1` — explicitly copies the
  mask into 3 channels, then thresholds.
- `selfie2.py`: `mask[..., None] > 0.1` — adds a size-1 channel axis and lets
  NumPy *broadcast* it across the 3 color channels automatically when it's
  compared against `image` in `np.where`.

Same result, less code. `selfie2.py` also converts only a *copy* of the frame
to RGB for MediaPipe and keeps the original BGR frame untouched for display —
`selfie.py` converts the same frame to RGB and back again, which is extra
work for the same outcome.

## Setup

### 1. Install Python 3.12

Download and install Python 3.12 from https://www.python.org/downloads/

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If that doesn't work, wipe the environment and reinstall from scratch:

```bash
deactivate
rm -rf .venv
```

Remake the virtual environment from Part 2, then install each piece by hand:

```bash
pip install "mediapipe==0.10.14" opencv-python numpy
pip install rembg
pip install -U "yt-dlp[default]"
pip install -U yt-dlp-ejs
pip install deno
```

### 4. Stop Here ✋. Make your choice: CPU or GPU for rembg

```bash
pip install "rembg[cpu]"   # for CPU — use this if you don't have an NVIDIA/CUDA GPU
pip install "rembg[gpu]"   # for NVIDIA/CUDA GPU
```

**What's the difference?** The model and the output are identical either way.
The only difference is the hardware. Background removal is mostly a huge
pile of multiply-and-add operations on pixel values. A CPU has a handful of fast,
general-purpose cores that work through those operations largely in sequence. A
GPU has thousands of small cores that do the *same* operation on thousands of
pixels simultaneously. That's a bad fit for most software, but a perfect fit for
neural networks. Therefore, the GPU build is typically several times faster per image.

The install flag picks which runtime gets installed underneath (`onnxruntime` vs.
`onnxruntime-gpu`). The GPU build only works if you actually have an NVIDIA card
with CUDA drivers; without one it will fail to load, so pick `[cpu]` if you're
unsure.

## Step 1: Background removal on a static image (rembg)

```bash
python segment.py
```

Reads `backgrounds/input_image.jpg`, removes the background, and saves the
result as `output_image.png` (PNG, not JPEG, because the output has a
transparent alpha channel that JPEG can't store). See the inline comments in
`segment.py` for exactly how the image array's shape changes at each step.

## Step 2: Background removal on live video (MediaPipe)

```bash
python selfie.py
```

Opens your webcam and replaces the background behind you with a solid
gray fill (`BG_COLOR = (192, 192, 192)` in `selfie.py`). Press Esc to quit.
Same basic idea as Step 1, but now running frame-by-frame on live video.

Look for 2 `CUSTOMIZE HERE` comments in `selfie.py`:

- **`BG_COLOR`**: any RGB tuple works. This
  [color list](https://inventwithpython.com/blog/complete-list-tkinter-colors-valid-and-tested.html)
  has names mapped to values if you don't want to guess numbers.
- **Background image**: swap the solid color for one of the
  `backgrounds/backgroundA.jpg` … `backgrounds/backgroundD.jpg` images
  already in this repo (or your own file) by uncommenting:

  ```python
  bg_image = cv2.imread('backgrounds/backgroundA.jpg')
  bg_image = cv2.resize(bg_image, (1920, 1080))
  ```

  This sits just before the `while` loop on purpose. You load and resize the
  image once, not every frame. Doing it inside the loop re-decodes the file
  from disk ~30 times a second, which makes higher-resolution files like
  `backgrounds/backgroundD.jpg` noticeably laggier.

  If your webcam resolution isn't 1920×1080 (width×height), change those two
  numbers to match. `image.shape` will tell you what your camera is actually
  capturing.

## Step 3: Movie background compositing (selfie2.py)

```bash
python selfie2.py
```

Same live segmentation as Step 2, but instead of a solid color, you're
composited onto a looping background *video*. Look for the `CUSTOMIZE HERE`
comment near the top of `selfie2.py` and set `BG_VIDEO_SOURCE` to either:

- a local video file path — a sample clip is already bundled at
  `backgrounds/video.mp4`, or
- a YouTube link (e.g. `https://www.youtube.com/watch?v=...`)

For YouTube links, `yt-dlp` downloads the clip to a temp file the first time
you run it, and OpenCV reads frames from that local file from then on, which
gets rid of CDN auth issues that break attempts to stream the YouTube URL
directly into OpenCV. The clip loops automatically once it reaches the end.
Press Esc to quit.

Notes:
- `model_selection=1` is used (same "landscape" model as the webcam section
  of `selfie.py`), since it holds up better for a person standing further
  back from the camera.
- Downloaded clips land in your OS temp directory (`/tmp` on macOS/Linux,
  `%TEMP%` on Windows) as `bg_clip.mp4`, and get overwritten each time
  `BG_VIDEO_SOURCE` points at a new link.
- YouTube links need two things beyond `yt-dlp` itself: `deno` (a JS runtime)
  and `yt-dlp-ejs` (the challenge-solver script that runs inside it to get
  past YouTube's current anti-bot check). Both are already in
  `requirements.txt`. If you see a warning about a missing JS runtime, or an
  error that says "challenge solving failed," double-check both installed
  correctly with `deno --version` and `pip show yt-dlp-ejs`.
- The retry logic (give up after `MAX_FAILED_FRAMES` = 30 dropped camera
  frames in a row) is the same safeguard used in `selfie.py` — it's there so
  a brief camera hiccup doesn't crash the script, but a genuinely
  disconnected camera still exits cleanly instead of hanging forever.

## Windows notes

- Webcam capture on Windows uses the DirectShow backend (`cv2.CAP_DSHOW`)
  automatically in both `selfie.py` and `selfie2.py` — faster camera
  startup than the default backend.
- Virtual environment binaries live in `.venv\Scripts\` on Windows (not
  `.venv/bin/` like macOS/Linux) — same idea, different folder name;
  activating the venv (`.venv\Scripts\activate`) puts the right folder on
  `PATH` either way.
- First-time webcam access on Windows may prompt a permissions dialog
  (Settings → Privacy & security → Camera) — allow it before running
  `selfie.py` or `selfie2.py`.
- Make sure you're running 64-bit Python (Windows: 64-bit Intel/AMD) —
  32-bit Python will fail to find prebuilt wheels for several of these
  packages.
