def load_image(filename, DATA_ROOT):
  image_data = {}
  source = cv2.imread(DATA_ROOT+"source/"+"source_"+filename) # source
  mask = cv2.imread(DATA_ROOT+"mask/"+"mask_"+filename) # mask
  target = cv2.imread(DATA_ROOT+"target/"+"target_"+filename) # target
  
  # normalize the images
  image_data['source'] = cv2.normalize(source.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['mask'] = cv2.normalize(mask.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['target'] = cv2.normalize(target.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['dims'] = target_offsets[int(filename[:-4])-1]
  
  return image_data

def display_image(image_data):
  # show the image
  mpplt.figure(figsize=(16,16))
  for i in range(3):
    if(i == 0):
      img_string = 'source'
    elif(i == 1):
      img_string = 'mask'
    else:
      img_string = 'target'
    img = image_data[img_string]
    mpplt.subplot(1,3,i+1)
    mpplt.imshow(img[:,:,[2,1,0]])
    