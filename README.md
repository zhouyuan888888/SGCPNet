#The Pytorch  Implementation of Real-time Semantic Segmentation via Spatial-detail Guided Context Propagation


###Before running please run
`python src/setup.py build_ext --build-lib=./src/.`

###For training 
`python src/train.py --evaluate False` 

###Testing
`python src/train.py --evaluate True --ckpt-path ./ckpt/cityscapes/checkpoint.pth.tar`  

###Calculate Parameters and FLOPs
` python src/parameter.py --weight w --height h`

