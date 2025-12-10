# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import numpy as np
import httpx
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading SAM model...")
sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95)
print("SAM loaded!")

def detect_walls(img):
    h, w = img.shape[:2]
    masks = mask_generator.generate(img)
    candidates = [(m["bbox"][0] + m["bbox"][2]/2, m["area"], m) for m in masks if m["area"] > 8000]
    candidates.sort(key=lambda x: x[1], reverse=True)

    walls = {}
    positions = {"left": False, "center": False, "right": False}

    for cx, area, m in candidates:
        if cx < w * 0.35 and not positions["left"]:
            pos = "left"
        elif cx > w * 0.65 and not positions["right"]:
            pos = "right"
        elif not positions["center"]:
            pos = "center"
        else:
            continue

        mask = m["segmentation"].astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        walls[pos] = mask
        positions[pos] = True
        if len(walls) == 3:
            break
    return walls

@app.post("/upload")
async def upload(file: UploadFile):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    walls = detect_walls(img)
    return {"walls": list(walls.keys())}

@app.post("/apply")
async def apply(
    room: UploadFile = File(...),
    wallpaper_left_url: str = Form(None),
    wallpaper_center_url: str = Form(None),
    wallpaper_right_url: str = Form(None),
):
    # Load room
    room_bytes = await room.read()
    room_img = cv2.imdecode(np.frombuffer(room_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = room_img.shape[:2]

    # Download wallpaper from URL (server-side → no CORS)
    async def download(url):
        if not url:
            return None
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=30.0)
            r.raise_for_status()
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return cv2.resize(img, (w, h)) if img is not None else None

    wp_left = await download(wallpaper_left_url)
    wp_center = await download(wallpaper_center_url)
    wp_right = await download(wallpaper_right_url)

    wallpapers = {"left": wp_left, "center": wp_center, "right": wp_right}
    masks = detect_walls(room_img)
    result = room_img.astype(np.float32)

    for pos, mask in masks.items():
        wp = wallpapers.get(pos)
        if wp is not None:
            mask3 = np.repeat(mask[:, :, None], 3, axis=2) / 255.0
            result = result * (1 - mask3) + wp.astype(np.float32) * mask3

    result = np.clip(result, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/jpeg")

@app.get("/")
def home():
    return {"status": "Magic View API Ready!"}