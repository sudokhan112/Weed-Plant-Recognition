# Weed-Plant-Recognition
This work use visible spectral-index based segmentation to segment the weeds from background. Mean, variance, kurtosis, and skewness are calculated for each input image and image quality (good or bad) is determined. Bad quality image is converted to good-quality image using contrast limited adaptive histogram equalization (CLAHE) before segmentation. The main objective of this chapter's work is to construct a redundant system which can segment weed and crop (green pixels) under varying outdoor condition.

Workflow:  

<img src="https://github.com/sudokhan112/Weed-Plant-Recognition/blob/main/Plant-Weed-Segmentation/fig5_1.png" width="600" height="600">

A good-quality image with histogram and statistics parametersfor R, G and B channels of the image.  
<img src="https://github.com/sudokhan112/Weed-Plant-Recognition/blob/main/Plant-Weed-Segmentation/fig5_4.png" width="600" height="400">

Original bad-quality image, bad-quality image after CLAHE,visible spectral-index based image segmentation (GREEN) on CLAHE appliedimage, histogram of original image, histogram of CLAHE image. Bad quality image is detected by statistics parameters and quality of image is improved by CLAHE.

<img src="https://github.com/sudokhan112/Weed-Plant-Recognition/blob/main/Plant-Weed-Segmentation/fig5_6.png" width="600" height="400">

Link to paper: https://asmedigitalcollection.asme.org/IMECE/proceedings-abstract/IMECE2019/59414/V004T05A042/1073161


**Cite this work**
```bash
@inproceedings{khan2019robust,
  title={Robust Weed Recognition Through Color Based Image Segmentation and Convolution Neural Network Based Classification},
  author={Khan, M Nazmuzzaman and Anwar, Sohel},
  booktitle={ASME International Mechanical Engineering Congress and Exposition},
  volume={59414},
  pages={V004T05A045},
  year={2019},
  organization={American Society of Mechanical Engineers}
}
```
