import poisson_blending as pbnd

# linear least squares solver
def linlsq_solver(A, b, dims):
  x = linalg.spsolve(A.tocsc(),b)
  return np.reshape(x,(dims[0],dims[1]))

# stitches poisson equation solution with target
def stitch_images(source, target, dims):
  target[dims[0]:dims[1], dims[2]:dims[3],:] = source
  return target

# performs poisson blending
def blend_image(data, BLEND_TYPE, GRAD_MIX):
  if(BLEND_TYPE == 1):
    image_solution = naive_copy(data)
    
  elif(BLEND_TYPE == 2):
    equation_param = []
    ch_data = {}
    
    # construct poisson equation 
    for ch in range(3):
      ch_data['source'] = data['source'][:,:,ch]
      ch_data['mask'] = data['mask'][:,:,ch]
      ch_data['target'] = data['target'][:,:,ch]
      ch_data['dims'] = data['dims']
      equation_param.append(pbnd.poisson_blending(ch_data, GRAD_MIX))

    # solve poisson equation
    image_solution = np.empty_like(data['source'])
    for i in range(3):
      image_solution[:,:,i] = linlsq_solver(equation_param[i][0],equation_param[i][1],data['source'].shape)
      
    image_solution = stitch_images(image_solution,image['target'],ch_data['dims'])
    
  else:
    # wrong option
    raise Exception('Wrong option! Available: 1. Naive, 2. Poisson')
      
  return image_solution
