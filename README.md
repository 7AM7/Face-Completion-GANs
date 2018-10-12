# Globally and Locally Consistent Image Completion
## Face-Completion-GANs

Keras implementation of ([Globally and Locally Consistent Image Completion](
http://hi.cs.waseda.ac.jp/%7Eiizuka/projects/completion/data/completion_sig2017.pdf)) on [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.  
![Alt text](result/network.JPG?raw=true "network")

## Requirements
* Opencv 3.4.3.18
* Keras 2.2.4

## Folder Setting
```
-data
  -img_align_celeba
    -img1.jpg
    -img2.jpg
    -...
```

## Train
```
$ python train.py 
```
## Test  
Download pretrained weights  
```
$ python test.py
```

## Results  
- ### [Kaggle kernal](https://www.kaggle.com/ahmedmoorsy/kernelf9d9e1f100?scriptVersionId=6384410)
- ### Python OpenCv
  ```
  Use your mouse to erase pixels in the image.  
  When you're done, press Q.  
  Result will be shown in few seconds.
  ```
![Alt text](result/1_test.png?raw=true "result")
![Alt text](result/2_test.png?raw=true "result")
![Alt text](result/3_test.png?raw=true "result")
![Alt text](result/4_test.png?raw=true "result")

   - ### After 10 Epochs
  ![Alt text](result/result_10.png?raw=true "result")
   - ### After 20 Epochs
  ![Alt text](result/result_20..png?raw=true "result")

## Feature works
  - ### using align the face landmarks with dlib before training.
