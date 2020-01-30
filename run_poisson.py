import numpy as np
import cv2
import matplotlib.pyplot as mpplt

# import src files
import load_image
import preprocess
import blend_image

def main_routine(IMAGE_NAME, DATA_ROOT, BLEND_TYPE=2, GRAD_MIX=False):
	target_offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], 
					  [-28, 88], [262,240], [315,629], [568,803], [378,806], [297,195]]
	BLEND_TYPE = BLEND_TYPE
	GRAD_MIX = GRAD_MIX

	image = load_image.load_image(IMAGE_NAME, DATA_ROOT)
	load_image.display_image(image) # plot data
	data = preprocess.preprocess(image)
	load_image.display_image(data) # plot for sanity check
	final_image = blend_image.blend_image(data, BLEND_TYPE, GRAD_MIX) # blend the image

	# plot results
	final_image = np.clip(final_image,0.0,1.0)
	mpplt.subplot(1,3,3)
	mpplt.imshow(final_image[:,:,[2,1,0]])

	# save image
	save_img = final_image * 255
	save_img = save_img.astype(np.uint8)
	cv2.imwrite(DATA_ROOT+'result/result_'+IMAGE_NAME, save_img, [cv2.IMWRITE_JPEG_QUALITY,90])

if __name__ == '__main__':
  print("Running Image Blending...")
  arglist = sys.argv.split() 
  IMAGE_NAME = arglist[1]
  DATA_ROOT = arglist[2]
  BLEND_TYPE = int(arglist[3])
  GRAD_MIX = bool(arglist[4])
  main_routine(IMAGE_NAME, DATA_ROOT, BLEND_TYPE, GRAD_MIX)
  