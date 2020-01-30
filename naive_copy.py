import numpy as np

# performs naive cut-paste from source to target
def naive_copy(image_data):
  # extract image data
  source = image_data['source']
  mask = image_data['mask']
  target = image_data['target']
  dims = image_data['dims']
  
  target[dims[0]:dims[1],dims[2]:dims[3],:] = target[dims[0]:dims[1],dims[2]:dims[3],:] * (1 - mask) + source * mask
  
  return target
  