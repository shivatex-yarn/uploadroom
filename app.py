# --------------------------------------------------------------
# app.py – Wallpaper Visualizer (MASK-ONLY, NO DIMMING)
# --------------------------------------------------------------
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

MAX_SIZE = 1200


def resize_to_match(base_img: Image.Image, img: Image.Image) -> Image.Image:
    """Resize img to match base_img size."""
    if img.size != base_img.size:
        return img.resize(base_img.size, Image.LANCZOS)
    return img


def tile_wallpaper(wallpaper_img: Image.Image, target_size: tuple) -> Image.Image:
    """Tile wallpaper to fill target size using Pillow."""
    wp_w, wp_h = wallpaper_img.size
    target_w, target_h = target_size

    tiled = Image.new("RGBA", target_size)
    for x in range(0, target_w, wp_w):
        for y in range(0, target_h, wp_h):
            tiled.paste(wallpaper_img, (x, y))

    return tiled.crop((0, 0, target_w, target_h))


def apply_wallpaper(template_img: Image.Image,
                    wallpaper_img: Image.Image,
                    mask_img: Image.Image) -> Image.Image:
    """
    Apply wallpaper only inside white regions of mask.
    - template_img: RGBA (room)
    - wallpaper_img: RGB (pattern)
    - mask_img: L (white = wall)
    """
    base = template_img.convert("RGBA")
    wp = wallpaper_img.convert("RGB")
    mask = mask_img.convert("L")

    # Resize everything to match
    wp = resize_to_match(base, wp)
    mask = resize_to_match(base, mask)

    # Tile wallpaper to cover full canvas
    tiled = tile_wallpaper(wp.convert("RGBA"), base.size)

    # --- Key Fix ---
    # We want wallpaper to appear *on white mask*,
    # keeping the rest of the room as-is.
    # So wallpaper = foreground, base = background.
    result = Image.composite(tiled, base, mask)

    return result


@app.route("/apply-wallpaper", methods=["POST"])
def apply():
    try:
        required = ["template", "wallpaper", "mask"]
        if not all(k in request.files for k in required):
            return jsonify({"error": "Missing template, wallpaper, or mask"}), 400

        template = Image.open(request.files["template"].stream).convert("RGBA")
        wallpaper = Image.open(request.files["wallpaper"].stream).convert("RGB")
        mask = Image.open(request.files["mask"].stream).convert("L")

        print(f"[DEBUG] Template: {template.size} | Wallpaper: {wallpaper.size} | Mask: {mask.size}")

        result = apply_wallpaper(template, wallpaper, mask)

        bio = io.BytesIO()
        result.save(bio, format="PNG")
        bio.seek(0)
        return send_file(bio, mimetype="image/png")

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("✅ Wallpaper backend running → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
