from yoloface.utils import post_process, get_outputs_names
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

import cv2
import numpy as np
import torch


CFG_PATH = "/content/drive/MyDrive/sample_data/yoloface/cfg/yolov3-face.cfg"
MODEL_WEIGHTS_PATH = "/content/drive/MyDrive/sample_data/model-weights/yolov3-wider_16000.weights"

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(CFG_PATH, MODEL_WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CONF_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

def get_image_vect(input_image_path):
  """
  Returns ndarray of cropped image.
  """
  print("Processing: ", input_image_path)
  frame = cv2.imread(input_image_path)

  # Create a 4D blob from a frame.
  blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
   [0, 0, 0], 1, crop=False)

  # Sets the input to the network
  net.setInput(blob)

  # Runs the forward pass to get output of the output layers
  outs = net.forward(get_outputs_names(net))

  # Remove the bounding boxes with low confidence
  faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

  # Ensure there is only 1 face in each image.
  if (len(faces) > 1):
    print("WARNING: More than 1 face detected. File: ", input_image_path)
  
  face = faces[0]
  
  left, top, width, height = face
  print(left, top, width, height)
  # Crop the image.
  cropped_frame = frame[top:top+height, left:left+width]
  print(cropped_frame.shape)
  # Flip the image to convert to RGB channel
  cropped_frame = np.flip(cropped_frame, axis=-1)
  # print(cropped_frame.shape)

  # PIL_image = Image.fromarray(np.uint8(cropped_frame)).convert('RGB')
  resized_frame = cv2.resize(cropped_frame, (160, 160), interpolation= cv2.INTER_LINEAR)

 

  # Create an inception resnet (in eval mode):
  resnet = InceptionResnetV1(pretrained='vggface2').eval()
 
  trans = transforms.Compose([transforms.ToTensor()])
  img_cropped_tensor = trans(resized_frame.copy())
  
  print(img_cropped_tensor.shape)
  img_embedding = resnet(img_cropped_tensor.unsqueeze(0))

  normalized_embedding = torch.nn.functional.normalize(img_embedding)
  return normalized_embedding.detach().numpy().flatten()
