# The Pytorch  Implementation of Real-time Semantic Segmentation via Spatial-detail Guided Context Propagation


### (i) Before running the scripts, please run `python src/setup.py build_ext --build-lib=./src/`.

### (ii) For the model training, please run the command `python src/train.py --evaluate False` .

### (iii) We provide the well-trained model in Google Drive for reproducing our results on the widely used semantic segmentation dataset, Cityscapes. Firstly, put the download checkpoint into the folder `./ckpt`. Secondly, please run the command `python src/test.py`, and the predictions for the test images are saved into the folder  `./ckpt/test_result`. At last,  run the command ` zip -r test_result.zip test_result`, and submit the `test_result.zip`  to the official online evaluator (`https://www.cityscapes-dataset.com/submit/`) 

