import os
import sys
import time
sys.path.append(os.path.dirname(__file__) + "/../")
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from scipy.misc import imread, imresize

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
from save_dict import save_dict_to_hdf5


cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
#import ipdb
# ipdb.set_trace()
# Read image from file
file_name = "image.png"
#image = imresize(imread(file_name, mode='RGB'), (384, 256))
image = imread(file_name, mode='RGB')
image_batch = data_to_input(image)

# Compute prediction with the CNN
for _ in range(5):
    now = time.time()
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    after = time.time()
    print(after - now)
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
# Visualise
print("POSE = \n")
print(list(map(list, pose)))
visualize.show_heatmaps(cfg, image, scmap, pose)
visualize.waitforbuttonpress()
