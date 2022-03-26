# The Pytorch  Implementation of Real-time Semantic Segmentation via Spatial-detail Guided Context Propagation

## Dataset Set
### *Camvid*
The Camvid dataset contains 701 images and 32 different semantic categories. We follow the previous works, and evaluate our segmentation model using 11 categories. For the details, please refer to the papers [1], [2], [3]. The Camvid dataset can be downloaded from [http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).     

### *Cityscapes*
The Cityscapes dataset contains 25 000 road scene images, in which 5000 images are finely annotated and 20 000 images are labelled with coarse annotation. In our experiments, we only adopt the fine-annotated subset. The fine-annotated subset involves 30 semantic categories. But we follow the previous works [2], [3], [4], [5], and adopt 19 categories in model evaluation.  The Cityscapes dataset can be downloaded from [http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/](https://www.cityscapes-dataset.com/). Accordingly, the toolkits for data pre-processing can be found at [https://github.com/mcordts/cityscapesScripts](https://github.com/mcordts/cityscapesScripts).

[1] Combining Appearance and Structure from Motion Features for Road Scene Understanding  
[2] ICNet for real-time semantic segmentation on high-resolution images  
[3] Efficient dense modules of asymmetric convolution for real-time semantic segmentation
[4] Pyramid scene parsing network
[5] PSANet: Point-wise spatial attention network for scene parsing

## Main Environment
python==3.6.2  
pytorch==1.1  
numpy==1.15  
torchvision==0.3.0  
pillow==7.1.2  
cython==0.29.20  
scipy==1.1.0  
scikit-learn==0.16.2  



##Script Running 
(1). Before running the scripts, please run `python src/setup.py build_ext --build-lib=./src/`.  

(2). For training the segmentation model, please run the command: `python src/train.py --evaluate [False/True]`.   

(3). We have saved the well-trained model on the Cityscapes dataset in [*Google Drive*](https://drive.google.com/drive/folders/1VuN_qSXjU3A1vQhT1JJH4PZbdJcHmdO2?usp=sharing) for reproducing our results. Specifically, firstly, put the download checkpoint into the folder `./ckpt`. Secondly, run the command `python src/test.py`. The predictions for the test images are saved into the folder  `./ckpt/test_result`. Thirdly, please  run the command ` zip -r test_result.zip test_result` in bash. Finally, submit the file `test_result.zip`  to the official online evaluator ([https://www.cityscapes-dataset.com/submit/](https://www.cityscapes-dataset.com/submit/)) to get the final performance in the Cityscapes' test set.

