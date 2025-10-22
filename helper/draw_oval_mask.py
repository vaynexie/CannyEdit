## code for interactively drawing an oval mask on the image

import cv2
import numpy as np
import os

points = []
drawing = False
img = None
window_name = 'Interactive Ellipse Mask'

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to record drawing trajectory"""
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text


def draw_oval_mask(image_path, save_dir="./mask_temp/", new_width=768, new_height=768):
    img = cv2.imread(image_path)
    new_size = (new_width, new_height)
    # Resize the image
    img = cv2.resize(img, new_size)
    img_file = image_path.split("/")[-1]
    img_name = img_file.split(".")[0]
    points.clear()

    # Create window and set mouse callback
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(new_width / 1.5), int(new_height / 1.5))
    instructions = "Press 'enter' to confirm a mask \nPress 'd' to save & close."
    cv2.setWindowTitle(window_name, "Mask Editor - Press 'Enter' to confirm | 'd' to save & close")
    cv2.setMouseCallback(window_name, mouse_callback)

    count_mask = 0
    while True:
        # Display real-time drawing
        display_img = img.copy()

        # Draw current trajectory
        if len(points) > 1:
            cv2.polylines(display_img, [np.array(points)], False, (0, 0, 255), 2)

        cv2.imshow(window_name, display_img)

        # Handle keyboard actions
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key to generate mask
            if len(points) >= 5:
                # Fit ellipse
                ellipse = cv2.fitEllipse(np.array(points))
                # Create blank mask
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                # Draw ellipse
                cv2.ellipse(mask, ellipse, 255, -1)
                # Show and save result
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                ind = len(os.listdir(save_dir))
                mask_file = f"mask_{ind}.png"
                mask_path = os.path.join(save_dir, mask_file)

                cv2.namedWindow('Generated Mask', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Generated Mask', int(new_width / 1.5), int(new_height / 1.5))
                cv2.imshow('Generated Mask', mask)
                cv2.imwrite(mask_path, mask)
                print(f"Mask has been saved to: {mask_path}")
                count_mask += 1
            else:
                print("Need at least five points to fit an oval mask")
        elif key == ord('r'):  # Reset drawing
            points.clear()
        elif key == ord('d'):  # Exit and save
            if points:
                pass
            else:
                print("Passed the image: ", img_file)
            break

    cv2.destroyAllWindows()
    return mask_path
