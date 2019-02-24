def preprocess(image_data):
  # extract image data
  source = image_data['source']
  mask = image_data['mask']
  target = image_data['target']
  
  # get image shape and offset
  Hs,Ws,_ = source.shape
  Ht,Wt,_ = target.shape
  Ho, Wo = image_data['dims']
  
  # adjust source and mask if offset is negative.
  # if mask is rolled eg. from the top it rolls 
  # to the bottom, crop the rolled portion
  if(Ho < 0):
    mask = np.roll(mask, Ho, axis=0)
    source = np.roll(source, Ho, axis=0)
    mask[Hs+Ho:,:,:] = 0 # added because Ho < 0
    source[Hs+Ho:,:,:] = 0
    Ho = 0
  if(Wo < 0):
    mask = np.roll(mask, Wo, axis=1)
    source = np.roll(source, Wo, axis=1)
    mask[:,Ws+Wo:,:] = 0
    source[:,Ws+Wo:,:] = 0
    Wo = 0
  
  # mask region on target
  H_min = Ho
  H_max = min(Ho + Hs, Ht)
  W_min = Wo
  W_max = min(Wo + Ws, Wt)
  
  # crop source and mask if they lie outside the bounds of the target
  source = source[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]
  mask = mask[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]
  
  return {'source':source, 'mask': mask, 'target': target, 'dims':[H_min,H_max,W_min,W_max]}
  