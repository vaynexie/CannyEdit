
## code for interactively selecting a point on the image

import cv2
import numpy as np
from typing import Optional, Tuple

def draw_star(img: np.ndarray, center: Tuple[int, int], size: int = 14, color=(0, 255, 255), thickness: int = 2):
    """
    Draw an 8-arm star centered at 'center' with given 'size'.
    """
    x, y = center
    # Horizontal and vertical
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    # Diagonals
    d = int(size * 0.7)
    cv2.line(img, (x - d, y - d), (x + d, y + d), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x - d, y + d), (x + d, y - d), color, thickness, cv2.LINE_AA)
    # Center dot
    cv2.circle(img, (x, y), max(1, thickness // 2 + 1), (0, 0, 0), -1, cv2.LINE_AA)

def select_single_point(
    image_path: str,
    new_width: int = 768,
    new_height: int = 768,
    star_size: int = 14,
    window_name: str = "Select one point (Left-click). Press D to confirm."
) -> Optional[Tuple[float, float]]:
    """
    GUI to select exactly one point on an image.
    - The image is resized to (new_width, new_height).
    - Left-click: set/replace the current point (only one stored).
    - D: confirm and return normalized (x, y) in [0, 1].
    - ESC: cancel and return None.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    resized = cv2.resize(img, (int(new_width), int(new_height)), interpolation=cv2.INTER_AREA)

    current_point = None  # (x, y) in resized coordinates

    instructions = [
        "Instructions:",
        " - Left-click: select/replace point",
        " - D: done/confirm",
        " - ESC: cancel",
        f"Image size: {new_width}x{new_height}"
    ]

    def render():
        canvas = resized.copy()
        # Draw star if a point exists
        if current_point is not None:
            draw_star(canvas, current_point, size=star_size, color=(0, 255, 255), thickness=2)

        # Draw overlays
        y0 = 20
        for line in instructions:
            cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            y0 += 18

        if current_point is not None:
            cx, cy = current_point
            info = f"Current: ({cx}, {cy})  Norm: ({cx/new_width:.3f}, {cy/new_height:.3f})"
            cv2.putText(canvas, info, (10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return canvas

    def on_mouse(event, x, y, flags, param):
        nonlocal current_point
        if event == cv2.EVENT_LBUTTONDOWN:
            # Clamp to bounds and set the single point
            x_clamped = int(np.clip(x, 0, new_width - 1))
            y_clamped = int(np.clip(y, 0, new_height - 1))
            current_point = (x_clamped, y_clamped)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, render())
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('d'), ord('D')):
            break
        if key == 27:  # ESC
            current_point = None
            break

    cv2.destroyWindow(window_name)

    if current_point is None:
        return None

    cx, cy = current_point
    x_norm = cx / float(new_width)
    y_norm = cy / float(new_height)
    return (x_norm, y_norm)


### Testcase below

# if __name__ == "__main__":

#
#     result = select_single_point(
#         image_path='./examples/bager1.png',
#         new_width=768,
#         new_height=768,
#         star_size=14,
#     )
#
#     if result is None:
#         print("No point selected.")
#     else:
#         x_norm, y_norm = result
#         print(f"Normalized point: ({x_norm:.6f}, {y_norm:.6f})")
