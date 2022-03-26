import os
import pdb
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = (os.path.split(curPath)[0])
sys.path.append(rootPath)

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import cv2
import torch.nn as nn
from thop import profile
from models.SGCPNet import SGCPNet

class ToTensor(object):
    def __call__(self, sample):
        image, mask= sample['image'], sample["mask"]
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), "mask":torch.from_numpy(mask)}

class Normalise(object):
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        return {'image': (self.scale * image - self.mean) / self.std, 'mask' : sample['mask']}


class CityscapesTest(Dataset):
    def __init__(self, data_file, data_dir, resolution, transform_test=None):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split(' '), datalist)]
        self.root_dir = data_dir
        self.transform_test = transform_test
        self.resolution = resolution

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))

        ##---resize for 1024x1024 input images---##
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image=np.array(image.resize(self.resolution))
        mask = np.array(mask.resize(self.resolution))

        sample = {'image': image, "mask": mask}
        sample = self.transform_test(sample)
        sample["file"] = img_name
        return sample

def main():
    num_classes = 19
    resolution = (2048, 1024)
    normalise_params = [1. / 255,  np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))]
    val_list = './data/test_list.txt'
    val_dir = './data/cityscapes/'
    composed_test = transforms.Compose([Normalise(*normalise_params), ToTensor()])
    load_path = "./ckpt/cityscapes.pth.tar"
    print(load_path)

    flops_1, params_1 = profile(SGCPNet(num_classes=num_classes), inputs=(torch.randn(1, 3, 2048, 1024),), report_missing=True)
    flops_2, params_2 = profile(SGCPNet(num_classes=num_classes), inputs=(torch.randn(1, 3, 1536, 768),), report_missing=True)
    print("1024 x 2048==> FLOPs:  {:.2f} G, Params:  {:.2f} M".format(flops_1/1e9, params_1/1e6))
    print("1536 x 768==> FLOPs:  {:.2f} G, Params:  {:.2f} M".format(flops_2 /1e9, params_2/1e6))

    pdb.set_trace()
    test_set = CityscapesTest(data_file=val_list, data_dir=val_dir, transform_test=composed_test, resolution=resolution)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    segmenter = nn.DataParallel(SGCPNet(num_classes=num_classes)).cuda()
    model_dict = torch.load(load_path)["segmenter"]
    segmenter.load_state_dict(model_dict)
    segmenter.eval()

    save_dir = "./ckpt/test_result/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            input = sample['image']
            input_var = torch.autograd.Variable(input).float().cuda()

            filename = sample['file'][0].split("/")[-1]
            filename = "_".join(filename.split("_")[:-1])+".png"
            save_pth = save_dir+filename
            output = segmenter(input_var)
            output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                 (2048, 1024), interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)

            output[output == 255] = 0
            output[output == 18] = 33
            output[output == 17] = 32
            output[output == 16] = 31
            output[output == 15] = 28
            output[output == 14] = 27
            output[output == 13] = 26
            output[output == 12] = 25
            output[output == 11] = 24
            output[output == 10] = 23
            output[output == 9] = 22
            output[output == 8] = 21
            output[output == 7] = 20
            output[output == 6] = 19
            output[output == 5] = 17
            output[output == 4] = 13
            output[output == 3] = 12
            output[output == 2] = 11
            output[output == 1] = 8
            output[output == 0] = 7
            cv2.imwrite(save_pth, output)

        print("Testing phrase finishes. Please zip the folder 'test_result', and submit to https://www.cityscapes-dataset.com/submit/ for offical online evaluation")

if __name__ == "__main__":
    main()