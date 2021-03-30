# Road_Crack_Detection

Automatic detection of pavement cracks is an important task in transportation maintenance for driving safety assurance. However, it remains a challenging task due to the intensity inhomogeneity of cracks and complexity of the background, e.g., the low contrast with surrounding pavement and possible shadows with similar intensity. Inspired by recent success on applying deep learning to computer vision and medical problems, a deep-learning based method for crack detection is proposed in this paper. A supervised deep convolutional neural network is trained to classify each image patch in the collected images. Quantitative evaluation conducted on a data set of 500 images of size 3264 χ 2448, collected by a low-cost smart phone, demonstrates that the learned deep features with the proposed deep learning framework provide superior crack detection performance when compared with features extracted with existing hand-craft methods.

Paper link : https://ieeexplore.ieee.org/document/7533052

---

Dataset for this work: https://data.mendeley.com/datasets/5y9wdsg2zt/1

he dataset is divided into two as negative and positive crack images for image classification. 
Each class has 20000images with a total of 40000 images with 227 x 227 pixels with RGB channels. 
The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). 

---

Sample of the Dataset

Positive:

<img src="https://github.com/Shakib-IO/Road_Crack_Detection/blob/main/Dataset/Positive/00460.jpg"> <img src="https://github.com/Shakib-IO/Road_Crack_Detection/blob/main/Dataset/Positive/00461.jpg"> <img src="https://github.com/Shakib-IO/Road_Crack_Detection/blob/main/Dataset/Positive/00462.jpg">

Negative:

<img src="https://github.com/Shakib-IO/Road_Crack_Detection/blob/main/Dataset/Negative/01684.jpg"> <img src="https://github.com/Shakib-IO/Road_Crack_Detection/blob/main/Dataset/Negative/01685.jpg"> <img src="https://github.com/Shakib-IO/Road_Crack_Detection/blob/main/Dataset/Negative/01686.jpg">
