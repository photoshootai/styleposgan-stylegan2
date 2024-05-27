# StylePoseGAN Implementation for Photoshoot
2020-2021 Startup Project where we were building a consumer app that generates completely new, high-quality photos of users for social media profiles. 

Our approach was based on a paper before its code was released: "Style and Pose Control for Image Synthesis of Humans from a Single Monocular View" Sarkar et al 2021.

In our implementation, we forked Lucidrains implementation of StyleGAN2. We also used Meta's DensePose through Detectron2.

## Project Overview
Taking good photos is hard. With the rising importance of having high-quality portraits and photos in today's digital world, we sought to provide users with photorealistic images of themselves for their digital profiles. We called this project Photoshoot AI (before several groups were using that name. 

In 2021, Sarkar et al. proposed a DL model known as _StyleposeGAN_. The theoretical assumption of this paper was that a portrait can be thought of as two components: 1) the pose, how the person is oriented in 3D space, and 2) the texture, how their appearance reflects their identity. See here:

![SPGAN_fig1](https://github.com/photoshootai/styleposgan-stylegan2/assets/41484082/41dfe52d-732b-45f8-a44a-76b888c863bb)
Screenshots from Sarkar 2021. A. Sample portrait B. "pose map" extracted by densepose (a DL model) C. Texture map extracted by a separate algorithm that models the images as a mesh and maps each body part to a specific part of the SMPL texture map.



