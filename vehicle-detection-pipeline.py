import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import csv
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Dropout
import sklearn
from keras.models import load_model
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from collections import deque

# load trained DNN
model = load_model("model.h5")
box_threshold = 0.5
heatmap_threshold = 3
heatmaps = deque(maxlen=3)

def slide_window(image, y_min=450, y_max=620):
    windows_list = []
    y = y_max
    while y >= y_min:
        box_size = 250 - int((y_max - y) * 1.2)
        for x in range(0, 1280 - box_size, int(box_size / 4)):
            #cv2.rectangle(image, (x, y), (x + box_size, y - box_size), (0, 0, 255), 2)
            windows_list.append(((x, y - box_size), (x + box_size, y)))
        y -= int(box_size / 3) + 20
    return windows_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 10)
    # Return the image
    return img



def process_image(image):
    # image size: 720x1280
    image_result = image.copy()
    image_positive_boxes = image.copy()
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    windows = slide_window(image)
    windows_positive_prediction = []
    for window in windows:
        image_region = image[window[0][1]:window[1][1], window[0][0]:window[1][0], :]

        image_region = cv2.resize(image_region, dsize=(64, 64))
        prediction = model.predict(image_region.reshape((1, 64, 64, 3)))
        if prediction >= box_threshold:
            cv2.rectangle(image_positive_boxes, window[0], window[1], (0, 0, 255), 2)
            windows_positive_prediction.append(window)

    # create heatmap
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, windows_positive_prediction)
    heatmaps.append(heatmap)
    # combine last heatmaps
    heatmaps_combined = sum(heatmaps)
    heatmaps_combined = apply_threshold(heatmaps_combined, heatmap_threshold)
    labels = label(heatmaps_combined)

    # draw boxes
    image_result = draw_labeled_bboxes(image_result, labels)

    return image_result




# process test image
test_image = mpimg.imread("./test_images/test6.jpg")
test_image_processed = process_image(test_image)
plt.imshow(test_image_processed)
plt.show()

# process video frame by frame
output_video = './output.mp4' # New video
clip1 = VideoFileClip('./project_video.mp4')#.subclip(4, 10) # Original video
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output_video, audio=False)