# Lights, Camera, Segment.py — A Computer Vision Workshop

A hands-on workshop teaching the basics of computer vision using background
segmentation as the running example. We walk through two different
segmentation approaches, in order of increasing complexity:

1. **rembg** — background removal on a single static image
2. **MediaPipe Selfie Segmentation** — the same idea, but applied live to a
   webcam video feed, to show that these models aren't limited to still images

## What this covers

- Installing Python and setting up an isolated environment for CV work
- Removing the background from a static image with `rembg`
- Choosing a CPU vs. GPU backend for a model, and why that choice matters
- Running a pretrained segmentation model (MediaPipe Selfie Segmentation) on
  live webcam input
- Swapping/customizing the background behind a person in real time

## Setup

### 1. Install Python 3.12

Download and install Python 3.12.0 for macOS from
https://www.python.org/downloads/macos/

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

If that doesn't work:

pip install "mediapipe==0.10.14" opencv-python "numpy<2"

4. 🛑 Pause here — choose CPU or GPU for rembg

rembg needs one more install, and which one depends on your hardware:

pip install "rembg[cpu]"   # for CPU — use this if you don't have an NVIDIA/CUDA GPU (e.g. any Mac)
pip install "rembg[gpu]"   # for NVIDIA/CUDA GPU

This is intentionally left out of requirements.txt — it's a good moment in
the workshop to talk about how the same model can run on different hardware
backends, and that most attendees (anyone on a Mac, no CUDA GPU) will want
rembg[cpu].

Step 1: Background removal on a static image (rembg)

python segment.py

Reads input_image.jpg, removes the background, and saves the result as
output_image.png (PNG, not JPEG, because the output has a transparent
alpha channel that JPEG can't store).

Step 2: Background removal on live video (MediaPipe)

python selfie.py

This opens your webcam and replaces the background behind you with a solid
gray fill (BG_COLOR = (192, 192, 192) in selfie.py). Press Esc to quit.
Same basic idea as Step 1, but now running frame-by-frame on live video.

Roadmap / things to add before the workshop

- [ ] Add backgroundA.jpeg, backgroundB.jpeg, backgroundC.jpeg so
attendees can swap in real image backgrounds instead of a solid color
(see the commented-out option in selfie.py under "The background can
be customized")
- [ ] Offer a few alternate solid-color schemes besides gray for people to
try (e.g. green screen, blue, black) by changing BG_COLOR

Notes

- model_selection=0 is used for static images (general model), and
model_selection=1 for the webcam loop (landscape model, better for
wider shots).
- MASK_COLOR / BG_COLOR are BGR tuples, not RGB.
- rembg.remove() returns an RGBA image — save as PNG, not JPEG, or flatten
it onto a background first if you need a .jpg.