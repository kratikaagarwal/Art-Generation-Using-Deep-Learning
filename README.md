<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center"> Art Generation Using Deep Learning</h3>

 


## About The Project
It is a Deep learning project which implements LA Gatys paper "A neural Algorithm of Artistic Style".

Neural Style Transfer is a fun and interesting technique which takes two images- a content image and  a style image and generates an output image which is actually the content image but painted in the style of the style image. Pre- trained convolutional neural network VGG19 is used to get feature representations of content and style image.

This technique of image style transfer is later extended to videos by applying style transfer to each frame of the video and then recombining the frames to get the stylized version of the video.

Pytorch is used at backend and Flask is used to deploy the model to web.
