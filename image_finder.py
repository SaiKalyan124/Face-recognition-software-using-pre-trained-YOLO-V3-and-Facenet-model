from pathlib import Path
import numpy as np
from image2vect import get_image_vect
import os

def calculate_image_vectors(images_path):
  images = Path(images_path).glob("*.jpg")
  image_strings = [str(p) for p in images]
  print(image_strings)

  # For each image, get vector
  #print(image_strings)
  print(image_strings)
  vect_dict = {}
  for i in image_strings:
    vect_dict[i] = get_image_vect(i)
  
  print(len(vect_dict))
  return vect_dict


def get_matching_images(input_file, threshold, vect_dict):
  """
  """
  # Get vector for input_file
  input_img_vect=get_image_vect(input_file)

  similar_images = []
  for image_path in vect_dict:
    vec = vect_dict[image_path]
    distance=eucledian_distance(input_img_vect, vec)
    if(distance < threshold):
      similar_images.append(os.path.basename(image_path))

  print(str(len(similar_images)) + " similar images found")
  return similar_images
  
def eucledian_distance(vect1, vect2):
  return np.linalg.norm(vect1 - vect2)
