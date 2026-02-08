"""Generate VoskTranscribe app icon."""

from PIL import Image, ImageDraw
import subprocess
import os

SIZE = 1024
CENTER = SIZE // 2

def make_icon():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background: rounded square with gradient-like blue
    pad = 40
    radius = 180
    # Dark navy background
    draw.rounded_rectangle(
        [pad, pad, SIZE - pad, SIZE - pad],
        radius=radius,
        fill=(20, 30, 70, 255),
    )
    # Subtle lighter overlay at top for depth
    draw.rounded_rectangle(
        [pad, pad, SIZE - pad, SIZE // 2],
        radius=radius,
        fill=(30, 50, 100, 80),
    )

    # --- Microphone body ---
    mic_w = 160
    mic_h = 280
    mic_top = 220
    mic_left = CENTER - mic_w // 2
    mic_right = CENTER + mic_w // 2
    mic_bottom = mic_top + mic_h

    # Mic capsule (rounded rect)
    draw.rounded_rectangle(
        [mic_left, mic_top, mic_right, mic_bottom],
        radius=mic_w // 2,
        fill=(255, 255, 255, 255),
    )

    # Mic grille lines
    for y in range(mic_top + 60, mic_bottom - 40, 30):
        draw.line(
            [(mic_left + 30, y), (mic_right - 30, y)],
            fill=(20, 30, 70, 80),
            width=4,
        )

    # --- Mic arc (U-shape cradle) ---
    arc_pad = 50
    arc_top = mic_top + 80
    arc_bottom = mic_bottom + 80
    arc_left = mic_left - arc_pad
    arc_right = mic_right + arc_pad
    draw.arc(
        [arc_left, arc_top, arc_right, arc_bottom + 40],
        start=0,
        end=180,
        fill=(255, 255, 255, 255),
        width=18,
    )

    # --- Mic stand (vertical line + base) ---
    stand_top = arc_bottom + 20
    stand_bottom = stand_top + 80
    draw.line(
        [(CENTER, stand_top), (CENTER, stand_bottom)],
        fill=(255, 255, 255, 255),
        width=18,
    )
    # Base
    base_w = 120
    draw.rounded_rectangle(
        [CENTER - base_w // 2, stand_bottom - 6, CENTER + base_w // 2, stand_bottom + 12],
        radius=6,
        fill=(255, 255, 255, 255),
    )

    # --- Sound waves (right side) ---
    wave_cx = mic_right + 80
    wave_cy = mic_top + mic_h // 2
    for i, r in enumerate([60, 110, 160]):
        alpha = 255 - i * 60
        color = (100, 180, 255, alpha)
        draw.arc(
            [wave_cx - r, wave_cy - r, wave_cx + r, wave_cy + r],
            start=-50,
            end=50,
            fill=color,
            width=14,
        )

    # --- Sound waves (left side, mirrored) ---
    wave_lx = mic_left - 80
    for i, r in enumerate([60, 110, 160]):
        alpha = 255 - i * 60
        color = (100, 180, 255, alpha)
        draw.arc(
            [wave_lx - r, wave_cy - r, wave_lx + r, wave_cy + r],
            start=130,
            end=230,
            fill=color,
            width=14,
        )

    # --- Text dots (bottom area, representing transcription) ---
    dot_y = stand_bottom + 60
    dot_colors = [(100, 180, 255), (140, 200, 255), (180, 220, 255)]
    for row in range(3):
        y = dot_y + row * 32
        num_dots = 8 - row * 2
        total_w = num_dots * 28 + (num_dots - 1) * 12
        start_x = CENTER - total_w // 2
        for d in range(num_dots):
            x = start_x + d * 40
            c = dot_colors[row % len(dot_colors)]
            draw.rounded_rectangle(
                [x, y, x + 28, y + 10],
                radius=5,
                fill=(*c, 200 - row * 40),
            )

    # Save as PNG
    png_path = os.path.join(os.path.dirname(__file__), "icon.png")
    img.save(png_path, "PNG")
    print(f"Saved {png_path}")

    # Convert to .icns using macOS sips + iconutil
    iconset_dir = os.path.join(os.path.dirname(__file__), "icon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512]
    for s in sizes:
        resized = img.resize((s, s), Image.LANCZOS)
        resized.save(os.path.join(iconset_dir, f"icon_{s}x{s}.png"))
        # @2x variant
        s2 = s * 2
        if s2 <= 1024:
            resized2 = img.resize((s2, s2), Image.LANCZOS)
            resized2.save(os.path.join(iconset_dir, f"icon_{s}x{s}@2x.png"))

    icns_path = os.path.join(os.path.dirname(__file__), "icon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", icns_path], check=True)
    print(f"Saved {icns_path}")

    # Cleanup iconset
    import shutil
    shutil.rmtree(iconset_dir)


if __name__ == "__main__":
    make_icon()
