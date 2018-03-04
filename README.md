# Vehicle Detection Project

### DNN architecture
I used a deep neural net (vehicle-detection-DNN.py). It has almost the same architecture as the NVDIA model I used in project 3. The difference is that I deleted the last convolutional layer because I trained the DNN with smaller images (64x64). The code for the training process is inside the file vehicle_detection_DNN.py. For preprocessing I added a lambda layer in Keras. This normalizes the inputs to values of -128 to +128. I wasn’t able to achieve good results with a normalization to -1 to +1. I also added a horizontally flipped version for each of the training images to help the DNN to generalize. Additionally, I changed the brightness randomly and shifted the image.
I also experimented with grayscale images but it didn’t work as well as RGB images.

### Sliding Window Search
#### Sclae an overlap of window search
The code is inside the slide_window() function in the file vehicle-detection-pipeline.py. My algorithm starts at the bottom of the interesting region (y_max = 620) with a box size of 200x200 pixels. The boxes overlap each other by 50%. It then increases the y value and decreases the box size by each iteration until it reaches the upper boundary (y_min = 450).
I chose this approach because it returned the regions where cars usually appear in the video.

#### Pipeline
At first, I fed all images determined by the slide_window() function into the DNN. These are all the positive classified boxes.
Then I created a heatmap.
The last step is to identify the cars with the scipy library.


### Video Implementation
I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.
Here's an example result showing the heatmap from a series of frames of video, the result of scipy.ndimage.measurements.label() and the bounding boxes then overlaid on the last frame of video:


### Discussion
#### Problems
I think I would need more training images to reduce the number of false positives. However, it would still fail in multiple scenarios:
1. It can only detect vehicles similar to the ones in the training sets. For example, it wouldn’t detect trucks or motorcycles.
2. It can’t detect cars which are far away. I would have to use smaller boxes. However, this would decrease the performance drastically and the DNN often fails to classify smaller images.
3. It can’t detect cars in the upper portion of the image.

