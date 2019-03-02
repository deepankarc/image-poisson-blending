# Poisson blending

Image blending is a process of transferring an image from the source domain to the target domain while ensuring the transformed pixels conform to the target domain to ensure consistency. Poisson Image Blending was introduced by Perez et al. to perform seamless blending of images. The idea utilizes the sensitivity of human observers to gradients in an image. By exploiting this we obtain a Poisson equation the solution to which yields the algorithm for seamless blending.  

This code implements gradient domain fusion of images via two techniques both of which are discussed in [1]:  

1. Poisson Blending  
2. Mixed Gradients  

The results are shown below:

![ScreenShot](/images/im1.jpg "Example 1 - Poisson Blending")  

![alt text](/images/im2.jpg "Example 2 - Mixed Gradients")  
<pre>
                                      Fig.1 - Image Blending </pre>
                                      
                                      

[1] - [P. Perez, M. Ganget, A. Blake - Poisson Image Editing](https://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)
