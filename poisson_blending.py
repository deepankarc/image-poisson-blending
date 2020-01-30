import numpy as np
import scipy.sparse as sps

def get_subimg(image, dims):
   return image[dims[0]:dims[1], dims[2]:dims[3]]

def poisson_blending(image, GRAD_MIX):
  # comparison function
  def _compare(val1, val2):
    if(abs(val1) > abs(val2)):
      return val1
    else:
      return val2
  
  # membrane (region where Poisson blending is performed)
  mask = image['mask']
  Hs,Ws = mask.shape
  num_pxls = Hs * Ws
  
  # source and target image
  source = image['source'].flatten(order='C')
  target_subimg = get_subimg(image['target'], image['dims']).flatten(order='C')

  # initialise the mask, guidance vector field and laplacian
  mask = mask.flatten(order='C')
  guidance_field = np.empty_like(mask)
  laplacian = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')

  for i in range(num_pxls):
    # construct the sparse laplacian block matrix
    # and guidance field for the membrane
    if(mask[i] > 0.99):
      
      laplacian[i, i] = 4
      
      # construct laplacian, and compute source and target gradient in mask
      if(i - Ws > 0):
        laplacian[i, i-Ws] = -1
        Np_up_s = source[i] - source[i-Ws]
        Np_up_t = target_subimg[i] - target_subimg[i-Ws]
      else:
        Np_up_s = source[i]
        Np_up_t = target_subimg[i]
        
      if(i % Ws != 0):
        laplacian[i, i-1] = -1
        Np_left_s = source[i] - source[i-1]
        Np_left_t = target_subimg[i] - target_subimg[i-1]
      else:
        Np_left_s = source[i]
        Np_left_t = target_subimg[i]
        
      if(i + Ws < num_pxls):
        laplacian[i, i+Ws] = -1
        Np_down_s = source[i] - source[i+Ws]
        Np_down_t = target_subimg[i] - target_subimg[i+Ws]
      else:
        Np_down_s = source[i]
        Np_down_t = target_subimg[i]
        
      if(i % Ws != Ws-1):
        laplacian[i, i+1] = -1
        Np_right_s = source[i] - source[i+1]
        Np_right_t = target_subimg[i] - target_subimg[i+1]
      else:
        Np_right_s = source[i]
        Np_right_t = target_subimg[i]
      
      # choose stronger gradient
      if(GRAD_MIX is False):
        Np_up_t = 0
        Np_left_t = 0
        Np_down_t = 0
        Np_right_t = 0
        
      guidance_field[i] = (_compare(Np_up_s, Np_up_t) + _compare(Np_left_s, Np_left_t) + 
                           _compare(Np_down_s, Np_down_t) + _compare(Np_right_s, Np_right_t))

    else:
      # if point lies outside membrane, copy target function
      laplacian[i, i] = 1
      guidance_field[i] = target_subimg[i]
  
  return [laplacian, guidance_field]
  