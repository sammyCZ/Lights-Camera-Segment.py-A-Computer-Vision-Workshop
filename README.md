# Lights, Camera, Segment.py — A Computer Vision Workshop

A hands-on workshop teaching the basics of computer vision using background
segmentation as the running example. We use two different approaches to the
same problem: Google's **MediaPipe Selfie Segmentation** model and **rembg**.

## What this covers

- Installing Python and setting up an isolated environment for CV work
- Running a pretrained segmentation model (MediaPipe Selfie Segmentation) on
  webcam input
- Swapping the background behind a person in real time
- (Separately) comparing results against `rembg` as a second segmentation approach

## Setup

### 1. Install Python 3.12

Download and install Python 3.12.0 for macOS from
https://www.python.org/downloads/macos/

This is needed to run the script built on MediaPipe's Selfie Segmentation
solution:
https://chuoling.github.io/mediapipe/solutions/selfie_segmentation.html#model_selection

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt 

if pip install -r requirements.txt doesn't work: 
pip install "mediapipe==0.10.14" opencv-python "numpy<2"

(A full requirements.txt with pinned versions is included in this repo —
pip install -r requirements.txt works too.)

4. Run the script

python selfie.py

This opens your webcam and replaces the background behind you with a solid
gray fill (BG_COLOR = (192, 192, 192) in selfie.py). Press Esc to quit.

Roadmap / things to add before the workshop

- [ ] Add backgroundA.jpeg, backgroundB.jpeg, backgroundC.jpeg so
attendees can swap in real image backgrounds instead of a solid color
(see the commented-out option in selfie.py under "The background can
be customized")
- [ ] Offer a few alternate solid-color schemes besides gray for people to
try (e.g. green screen, blue, black) by changing BG_COLOR
- [ ] Add the rembg comparison example

Notes

- model_selection=0 is used for static images (general model), and
model_selection=1 for the webcam loop (landscape model, better for
wider shots).
- MASK_COLOR / BG_COLOR are BGR tuples, not RGB.