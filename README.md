# StylePoseGAN Implementation for Photoshoot
2020-2021 Startup Project where we were building a consumer app that generates completely new, high-quality photos of users for social media profiles. 

Our approach was based on a paper before its code was released: "Style and Pose Control for Image Synthesis of Humans from a Single Monocular View" Sarkar et al 2021.

In our implementation, we forked Lucidrains implementation of StyleGAN2. We also used Meta's DensePose through Detectron2.

## Project Overview
Taking good photos is hard. With the rising importance of having high-quality (HQ) portraits and photos in today's digital world, we sought to provide users with photorealistic images of themselves for their digital profiles. We called this project Photoshoot AI (before several groups were using that name. 

In 2021, Sarkar et al. proposed a DL model known as _StyleposeGAN_. The theoretical assumption of this paper was that a portrait can be thought of as two components: 1) the pose, how the person is oriented in 3D space, and 2) the texture, how their appearance reflects their identity. See here:

**Portraits consist of Pose and Texture**
![SPGAN_fig1 drawio](https://github.com/photoshootai/styleposgan-stylegan2/assets/41484082/ad150641-61eb-4303-87b7-beed62659a73)
Screenshots from Sarkar 2021. A. Sample portrait B. "pose map" extracted by densepose (a DL model) C. Texture map extracted by a separate algorithm that models the images as a mesh and maps each body part to a specific part of the SMPL texture map.

**StyleposeGAN combines Pose and Texture to Reconstruct original Image**
![SPGAN_architecture drawio](https://github.com/photoshootai/styleposgan-stylegan2/assets/41484082/a4ca8721-e4fb-47c4-b837-ff861e10878a)
StyleposeGAN Architecture. 
A. Densepose Map is fed into a CNN feature extractor "PNet" which encodes the pose into a 3D Tensor.
B. SMPL Texture Map is fed into a separate CNN feature extractor "ANet" which encodes the appearance (texture) into a vector.
C. The core functionality of StyleposeGAN is GNet, a generative adversarial network (GAN) that aims to reconstruct the original image. Quality is enforced by a discriminator network, while identity is maintained with a face loss, L1 and VGG loss both enforce the model to pose the person correctly. 

 #### Our product idea was to take in users' texture map, but replace their pose with that of a professional model. At the click of a button, our app would choose a random pose and the network would then produce an image of the user, posed like a model. 

