from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import cv2
import numpy as np
import logging
from pathlib import Path
import shutil
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import traceback
import uuid

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
UPLOAD_FOLDER = Path("uploads")
MASK_FOLDER = Path("masks")
WALLS_FOLDER = Path("walls")
RESULTS_FOLDER = Path("results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

for p in (UPLOAD_FOLDER, MASK_FOLDER, WALLS_FOLDER, RESULTS_FOLDER):
    p.mkdir(parents=True, exist_ok=True)

# Mount static directories
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/masks", StaticFiles(directory=MASK_FOLDER), name="masks")
app.mount("/walls", StaticFiles(directory=WALLS_FOLDER), name="walls")
app.mount("/results", StaticFiles(directory=RESULTS_FOLDER), name="results")

# Load SAM model
sam_checkpoint = Path("checkpoints/sam_vit_h_4b8939.pth")
model_type = "vit_h"

if not sam_checkpoint.exists():
    raise FileNotFoundError(
        f"SAM checkpoint not found at {sam_checkpoint}. "
        "Download from https://github.com/facebookresearch/segment-anything"
    )

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
auto_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    min_mask_region_area=1000,
)
logger.info(f"SAM model loaded from {sam_checkpoint.name}")


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Wall Detection Functions ---
def detect_individual_walls(image_rgb, mask_preds):
    h, w, _ = image_rgb.shape
    wall_masks = []
    mask_preds.sort(key=lambda x: x['area'], reverse=True)
    position_taken = {"left": False, "right": False, "center": False}

    for i, mask_obj in enumerate(mask_preds):
        mask = mask_obj['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, mask_w, mask_h = cv2.boundingRect(contour)
        aspect = mask_w / mask_h if mask_h > 0 else 0
        area_ratio = (mask_obj.get('area', mask.sum())) / (h * w)
        is_large_enough = area_ratio > 0.05
        is_tall = aspect < 3.0
        is_vertical = mask_h > h * 0.25

        if is_large_enough and is_tall and is_vertical:
            wall_center_x = x + mask_w / 2
            wall_position = (
                "left" if wall_center_x < w / 3
                else "right" if wall_center_x > 2 * w / 3
                else "center"
            )
            if not position_taken[wall_position]:
                wall_masks.append({
                    'id': f"wall_{i}",
                    'position': wall_position,
                    'mask': (mask * 255).astype(np.uint8),
                    'bbox': [int(x), int(y), int(mask_w), int(mask_h)],
                    'area': int(mask.sum())
                })
                position_taken[wall_position] = True
                logger.info(f"Selected {wall_position} wall (area_ratio={area_ratio:.3f})")
        if all(position_taken.values()):
            break

    return wall_masks


def create_combined_mask(wall_masks, resized_shape):
    combined_mask = np.zeros(resized_shape[:2], dtype=np.uint8)
    for wall in wall_masks:
        if wall['mask'].shape != combined_mask.shape:
            wall_mask_resized = cv2.resize(
                wall['mask'], (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        else:
            wall_mask_resized = wall['mask']
        combined_mask = cv2.bitwise_or(combined_mask, wall_mask_resized)
    return combined_mask


async def process_room_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    try:
        image_enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=15)
        max_size = 512
        h_orig, w_orig = image.shape[:2]
        scale = 1.0
        if max(h_orig, w_orig) > max_size:
            scale = max_size / max(h_orig, w_orig)
            image_resized = cv2.resize(image_enhanced, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        else:
            image_resized = image_enhanced.copy()
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        masks = auto_generator.generate(image_rgb)
        logger.info(f"Generated {len(masks)} masks from SAM (resized size={image_rgb.shape[:2]})")

        if not masks:
            raise ValueError("No masks detected in the image")

        wall_masks_resized = detect_individual_walls(image_rgb, masks)
        if not wall_masks_resized:
            raise ValueError("No walls detected in the image")

        processed_walls = []
        for i, wmask in enumerate(wall_masks_resized):
            if scale != 1.0:
                mask_back = cv2.resize(wmask['mask'], (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            else:
                mask_back = wmask['mask']
            kernel = np.ones((5, 5), np.uint8)
            mask_back = cv2.morphologyEx(mask_back, cv2.MORPH_CLOSE, kernel)
            mask_back = cv2.morphologyEx(mask_back, cv2.MORPH_OPEN, kernel)
            uid = uuid.uuid4().hex[:8]
            wall_filename = f"wall_{wmask['position']}_{image_path.stem}_{uid}.png"
            wall_path = WALLS_FOLDER / wall_filename
            success = cv2.imwrite(str(wall_path), mask_back)
            if not success:
                raise ValueError(f"Failed to save wall mask at {wall_path}")

            processed_walls.append({
                'id': wmask['id'],
                'position': wmask['position'],
                'name': f"{wmask['position'].title()} Wall",
                'maskURL': f"/walls/{wall_filename}",
                'selected': wmask['position'] == 'center',
                'mask_array': mask_back,
                'bbox': wmask['bbox']  # Include bbox in processed_walls
            })

        combined_mask_resized = create_combined_mask(wall_masks_resized, image_rgb.shape)
        combined_mask_filename = f"mask_{image_path.stem}_{uuid.uuid4().hex[:8]}.png"
        combined_mask_path = MASK_FOLDER / combined_mask_filename
        cv2.imwrite(str(combined_mask_path), combined_mask_resized)

        response_walls = []
        for pw in processed_walls:
            response_walls.append({
                'id': pw['id'],
                'position': pw['position'],
                'name': pw['name'],
                'maskURL': pw['maskURL'],
                'selected': pw['selected'],
                'bbox': pw['bbox']  # Add bbox to response
            })

        return response_walls, combined_mask_filename, processed_walls
    except Exception as e:
        logger.error(f"Error in process_room_image: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


def apply_wallpaper_to_walls(room_img, wall_masks, wallpapers, alpha=0.85):
    h, w = room_img.shape[:2]
    result = room_img.copy().astype(np.float32)
    wallpaper_dict = {wp['position']: wp['image'] for wp in wallpapers if wp.get('image') is not None}

    for wall in wall_masks:
        mask = wall.get('mask')
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_float = (mask.astype(np.float32) / 255.0)[:, :, None]
        wallpaper = wallpaper_dict.get(wall['position'], None)
        if wallpaper is None:
            continue
        wallpaper_resized = cv2.resize(wallpaper, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        mask_alpha = mask_float * alpha
        inv_alpha = 1.0 - mask_alpha
        result = result * inv_alpha + wallpaper_resized * mask_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


# --- API Endpoints ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    filename_safe = Path(file.filename).name
    if not allowed_file(filename_safe):
        raise HTTPException(status_code=400, detail="Invalid file type")
    uid = uuid.uuid4().hex[:8]
    filename = f"{Path(filename_safe).stem}_{uid}{Path(filename_safe).suffix}"
    img_path = UPLOAD_FOLDER / filename

    try:
        with img_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        response_walls, combined_mask_filename, _ = await process_room_image(img_path)
        return {
            "original": f"/uploads/{filename}",
            "mask": f"/masks/{combined_mask_filename}",
            "walls": response_walls
        }
    except Exception as e:
        if img_path.exists():
            img_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@app.post("/apply_wallpaper")
async def apply_wallpaper(
    room: UploadFile = File(...),
    wallpaper_left: UploadFile = File(default=None),
    wallpaper_right: UploadFile = File(default=None),
    wallpaper_center: UploadFile = File(default=None)
):
    logger.info("Received /apply_wallpaper request")  # Debug log
    if not room:
        raise HTTPException(status_code=400, detail="Room image required")
    room_filename_safe = Path(room.filename).name
    if not allowed_file(room_filename_safe):
        raise HTTPException(status_code=400, detail="Invalid room file type")
    room_uid = uuid.uuid4().hex[:8]
    room_filename = f"{Path(room_filename_safe).stem}_{room_uid}{Path(room_filename_safe).suffix}"
    room_path = UPLOAD_FOLDER / room_filename

    try:
        with room_path.open("wb") as buffer:
            shutil.copyfileobj(room.file, buffer)
        response_walls, _, processed_walls_internal = await process_room_image(room_path)

        wallpapers = []
        for pos, wp_file in [('left', wallpaper_left), ('right', wallpaper_right), ('center', wallpaper_center)]:
            if wp_file:
                wp_name_safe = Path(wp_file.filename).name
                if not allowed_file(wp_name_safe):
                    raise HTTPException(status_code=400, detail=f"Invalid wallpaper file for {pos}")
                wp_uid = uuid.uuid4().hex[:8]
                wp_filename = f"{Path(wp_name_safe).stem}_{wp_uid}{Path(wp_name_safe).suffix}"
                wp_path = UPLOAD_FOLDER / wp_filename
                with wp_path.open("wb") as buffer:
                    shutil.copyfileobj(wp_file.file, buffer)
                wp_img = cv2.imread(str(wp_path))
                wallpapers.append({'position': pos, 'image': wp_img})
                logger.info(f"Processed wallpaper for {pos}")

        if not wallpapers:
            raise HTTPException(status_code=400, detail="No wallpapers provided")

        wall_masks_for_apply = [{'position': pw['position'], 'mask': pw['mask_array']} for pw in processed_walls_internal]
        room_img = cv2.imread(str(room_path))
        result_img = apply_wallpaper_to_walls(room_img, wall_masks_for_apply, wallpapers, alpha=0.95)

        result_uid = uuid.uuid4().hex[:8]
        result_filename = f"result_{Path(room_filename).stem}_{result_uid}{Path(room_filename).suffix}"
        result_path = RESULTS_FOLDER / result_filename
        cv2.imwrite(str(result_path), result_img)
        logger.info(f"Result saved at {result_path}")
        return {"result": f"/results/{result_filename}"}
    except Exception as e:
        logger.error(f"Error in /apply_wallpaper: {str(e)}\n{traceback.format_exc()}")
        if room_path.exists():
            room_path.unlink()
        for f in (wallpaper_left, wallpaper_right, wallpaper_center):
            if f and (UPLOAD_FOLDER / f.filename).exists():
                (UPLOAD_FOLDER / f.filename).unlink()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await room.close()
        for f in (wallpaper_left, wallpaper_right, wallpaper_center):
            if f:
                await f.close()


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)