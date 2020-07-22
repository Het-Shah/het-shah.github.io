---
layout: page
title: Texture Segmentation
img: /assets/img/texture_proj.png
---
The project was done in the summer of 2019 as a Summer Research Intern at <a href="https://bisag.gujarat.gov.in/">Bhaskarachar Institute of Space Applications and Geo-Informatics</a>.

The project aims to correctly mapping a particular type of farming technique known as horticulture from a given satellite image. The model that was made predicts a binary mask for the same. Given a 256\*256 image, which is taken from the main satellite image, of grayscale the model gives out a binary mask of size 256\*256 which we can resize the original image dimensions and make the whole mask for the satellite image that was given for the prediction. 

The project uses a deep learning technique known as Semantic Segmentation. The model that the project uses is a U-Net model. This model is often used on Segmentation of Medical images. As the dataset which was even similar to the task given wasn’t available publicly so the dataset was made by us. We used an open source software, ‘labelme’, to help us label the dataset.

<div class="social">
  <span class="contacticon center">
    <a href="https://www.github.com/Het-Shah/Texture-segmentation" target="_blank" title="GitHub"><i class="fab fa-github"></i></a>
    <a href="https://drive.google.com/file/d/1Db44TbpPOwdzQ7xUp8GuhXcMqBS2eUCZ/view?usp=sharing" target="_blank" title="Report"><i class="fas fa-file-alt"></i></a>
  </span>
  <div class="col three caption">
    Github Link to the project and Report.
  </div>
</div>