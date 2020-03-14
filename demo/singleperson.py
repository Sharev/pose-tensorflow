import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from skimage import io

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
from ludwig import visualize
from numpy import array

cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = "demo/image.png"
image = io.imread(file_name)

image_batch = data_to_input(image)

#imageChanged = array(image_batch).reshape(1, 512,274,3)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

# Visualise
visualize.show_heatmaps(cfg, image, scmap, pose)
visualize.waitforbuttonpress()
