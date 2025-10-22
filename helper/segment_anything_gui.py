
## code for interactively selecting an object mask from image via SAM2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib as mpl
from helper.sam_utils import SAM
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import os
from matplotlib.patches import Circle
import copy
import gc
import matplotlib


def run_gui(img_input_filepath, output_dir="./mask_temp/", new_width=768, new_height=768,
            sam_checkpoint=None):
    """
    Initializes and runs the interactive segmentation GUI.

    The key change is retrieving the output path *after* plt.show() completes,
    ensuring the GUI has been used and the path has been set.
    """
    img = cv2.imread(img_input_filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image file not found at: {img_input_filepath}")

    new_size = (new_width, new_height)
    img = cv2.resize(img, new_size)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    # Check if the SAM checkpoint file exists
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint file not found: {sam_checkpoint}. Please download it.")

    # Step 1: Create the Segmenter instance.
    segmenter = Segmenter(img, output_dir=output_dir, sam_checkpoint=sam_checkpoint)

    # Step 2: Show the GUI. This is a blocking call.
    # The script will pause here until the user closes the Matplotlib window.
    plt.show(block=True)

    # Step 3: *After* the window is closed, get the path.
    # The `save_annotation` method now correctly returns the path that was set
    # when the user pressed 'Escape'.
    output_path = segmenter.save_annotation()

    # print("Interactive session finished.")
    # if output_path:
    #     print(f"Returned mask path: {output_path}")
    # else:
    #     print("No mask was saved (window may have been closed without saving).")

    # Step 4: Clean up resources.
    del segmenter
    torch.cuda.empty_cache()
    gc.collect()

    return output_path


class Segmenter():
    def __init__(self, img, output_dir, sam_checkpoint):
        self.img = img
        self.output_dir = output_dir
        self.sam_checkpoint = sam_checkpoint
        self.min_mask_region_area = 500
        # This is initialized to None. It only gets a value when the user saves.
        self.save_pathh = None

        ## SAM2
        self.sam = SAM()
        model_type = "sam2.1_hiera_large"
        self.sam.build_model(model_type, self.sam_checkpoint,device="cuda")
        self.sam.predictor.set_image(self.img)

        self.color_set = set()
        self.current_color = self.pick_color()
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []

        self.fig, self.ax = plt.subplots(figsize=(10 * (self.img.shape[1] / max(self.img.shape)),
                                                  10 * (self.img.shape[0] / max(self.img.shape))))
        self.fig.suptitle('Segment Anything GUI', fontsize=26)
        self.ax.set_title("Press 'h' for help, Press 'enter' to confirm a mask \n Press 'd' to save & close. The mask is union of all masks obtained here. \n Press 'r' to save & close. The mask is the inverse of union of all masks obtained here."  , fontsize=12)
        self.im = self.ax.imshow(self.img, cmap=mpl.cm.gray)
        self.ax.autoscale(False)
        self.label = 1
        self.add_plot, = self.ax.plot([], [], 'o', markerfacecolor='green', markeredgecolor='black', markersize=5)
        self.rem_plot, = self.ax.plot([], [], 'x', markerfacecolor='red', markeredgecolor='red', markersize=5)
        self.mask_data = np.zeros((self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8)
        for i in range(3):
            self.mask_data[:, :, i] = self.current_color[i]
        self.mask_plot = self.ax.imshow(self.mask_data)
        self.prev_mask_data = np.zeros((self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8)
        self.prev_mask_plot = self.ax.imshow(self.prev_mask_data)
        self.contour_plot, = self.ax.plot([], [], color='black')
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.show_help_text = False
        self.help_text = plt.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center',
                                  transform=self.ax.transAxes, fontsize=12)
        self.opacity = 120
        self.global_masks = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        self.last_mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=bool)
        self.full_legend = []
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    def save_annotation(self):
        # This function simply returns the current state of self.save_pathh
        return self.save_pathh

    def pick_color(self):
        while True:
            color = tuple(np.random.randint(low=50, high=255, size=3).tolist())  # Avoid dark colors
            if color not in self.color_set:
                self.color_set.add(color)
                return color

    def _on_key(self, event):
        if event.key == 'z':
            self.undo()
        elif event.key == 'enter':
            self.new_tow()
        elif event.key == 'd':
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Generate a unique filename
            ind = len(os.listdir(self.output_dir))
            mask_file = f"mask_{ind}.png"
            mask_path = os.path.join(self.output_dir, mask_file)

            # Create the final binary mask and save it
            mask_output = np.where(self.global_masks != 0, 255, 0).astype(np.uint8)
            cv2.imwrite(mask_path, mask_output)

            # This is the crucial step: update the instance variable
            self.save_pathh = mask_path
            #print(f"Mask saved to {self.save_pathh}")

            # Close the window, which allows plt.show() to return
            plt.close(self.fig)
        elif event.key == 'r':
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Generate a unique filename
            ind = len(os.listdir(self.output_dir))
            mask_file = f"mask_{ind}_inverse.png"
            mask_path = os.path.join(self.output_dir, mask_file)

            # Create the inverse binary mask and save it
            mask_output = np.where(self.global_masks == 0, 255, 0).astype(np.uint8)
            cv2.imwrite(mask_path, mask_output)

            # Update the instance variable
            self.save_pathh = mask_path
            # print(f"Inverse mask saved to {self.save_pathh}")

            # Close the window, which allows plt.show() to return
            plt.close(self.fig)

        elif event.key == 'h':
            if not self.show_help_text:
                self.help_text.set_text("• 'left click': select a point inside object\n"
                                        "• 'right click': select a point to exclude\n"
                                        "• 'enter': confirm current mask and start new one\n"
                                        "• 'z': undo last point or last confirmed mask\n"
                                        "• 'd': save the union of all masks and close, then exit\n"
                                        "• 'r': save the inverse of the union of all masks and close, then exit")
                self.help_text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='black'))
                self.show_help_text = True
            else:
                self.help_text.set_text('')
                self.show_help_text = False
            self.fig.canvas.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax or (event.button not in [1, 3]): return
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))

        if event.button == 1:
            self.trace.append(True)
            self.add_xs.append(x)
            self.add_ys.append(y)
            self.show_points(self.add_plot, self.add_xs, self.add_ys)
        else:
            self.trace.append(False)
            self.rem_xs.append(x)
            self.rem_ys.append(y)
            self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

        self.get_mask()

    def show_points(self, plot, xs, ys):
        plot.set_data(xs, ys)
        self.fig.canvas.draw()

    def clear_mask(self):
        self.contour_plot.set_data([], [])
        self.mask_data.fill(0)
        for i in range(3):  # Re-apply current color
            self.mask_data[:, :, i] = self.current_color[i]
        self.mask_plot.set_data(self.mask_data)
        self.fig.canvas.draw()

    def get_mask(self):
        if not self.add_xs and not self.rem_xs:
            self.clear_mask()
            return

        mask, _, _ = self.sam.predictor.predict(
            point_coords=np.array(list(zip(self.add_xs, self.add_ys)) + list(zip(self.rem_xs, self.rem_ys))),
            point_labels=np.array([1] * len(self.add_xs) + [0] * len(self.rem_xs)),
            multimask_output=False
        )
        mask = mask[0].astype(np.uint8)

        mask[self.global_masks > 0] = 0
        mask = self.remove_small_regions(mask, self.min_mask_region_area, "holes")
        mask = self.remove_small_regions(mask, self.min_mask_region_area, "islands")

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        xs, ys = [], []
        for contour in contours:
            xs.extend(contour[:, 0, 0].tolist() + [np.nan])
            ys.extend(contour[:, 0, 1].tolist() + [np.nan])
        self.contour_plot.set_data(xs, ys)

        self.mask_data[:, :, 3] = mask * self.opacity
        self.mask_plot.set_data(self.mask_data)
        self.fig.canvas.draw()

    def undo(self):
        if len(self.trace) == 0:
            if np.any(self.last_mask):
                self.global_masks[self.last_mask] = 0
                self.prev_mask_data[:, :, 3][self.last_mask] = 0
                self.prev_mask_plot.set_data(self.prev_mask_data)
                self.last_mask.fill(False)  # Prevent multiple undos of same mask
                self.label -= 1
                if self.full_legend: self.full_legend.pop()
                self.ax.legend(handles=self.full_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
                self.clear_mask()
                #print("Last confirmed mask removed.")
        else:
            if self.trace.pop():
                self.add_xs.pop()
                self.add_ys.pop()
                self.show_points(self.add_plot, self.add_xs, self.add_ys)
            else:
                self.rem_xs.pop()
                self.rem_ys.pop()
                self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)
            self.get_mask()

    def new_tow(self):
        mask = self.mask_data[:, :, 3] > 0
        if not np.any(mask):
            #print("No mask to confirm. Please select points first.")
            return

        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []
        self.show_points(self.add_plot, self.add_xs, self.add_ys)
        self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

        self.global_masks[mask] = self.label
        self.last_mask = mask.copy()

        self.prev_mask_data[:, :, :3][mask] = self.current_color
        self.prev_mask_data[:, :, 3][mask] = 255
        self.prev_mask_plot.set_data(self.prev_mask_data)

        self.full_legend.append(Circle((0, 0), 1, color=np.array(self.current_color) / 255, label=f'Mask {self.label}'))
        self.ax.legend(handles=self.full_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

        self.current_color = self.pick_color()
        self.label += 1

        self.clear_mask()


    @staticmethod
    def remove_small_regions(mask, area_thresh, mode):
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0: return mask
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask
