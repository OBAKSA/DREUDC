import cv2
import numpy as np
import torch
import argparse
from metrics import *
from archs.unet import UNet_raw2rgb
from archs.awnet import AWNet
from archs.pynet import PyNET

import torch.nn as nn
from torch.utils.data import DataLoader
from data import ImageDataset_raw2rgb

device = torch.device('cuda')

data_path = './dataset/'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='deunet', help='type of model')
parser.add_argument('--panel_type', type=str, default='axon', help='type of dataset')

opt = parser.parse_args()


def load_models(model_type, panel_type):
    ### define model ###
    if model_type == 'deunet':
        model = UNet_raw2rgb()
    elif model_type == 'awnet':
        model = AWNet()
    elif model_type == 'pynet':
        model = PyNET()

    else:
        print('Unidentified model type, please check.')

    model.to(device)

    ### define panel type ###
    if panel_type == 'axon':
        axon_path = './pretrained_models/' + opt.model_type + '/axon.pth'
        checkpoint = torch.load(axon_path, map_location='cuda')
        model.load_state_dict(checkpoint)

    elif panel_type == 'zfold':
        zfold_path = './pretrained_models/' + opt.model_type + '/zfold.pth'
        checkpoint = torch.load(zfold_path, map_location='cuda')
        model.load_state_dict(checkpoint)

    else:
        print('Unidentified panel type, please check.')

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


### initialize test & performance metrics ###

test_psnr = []
test_ssim = []


def test(test_data_path, model_type, panel_type):
    # load dataset
    test_dataset = ImageDataset_raw2rgb(image_dirs=test_data_path, panel_type=panel_type)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # load model
    model = load_models(model_type=model_type, panel_type=panel_type)

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            udc_image = data[0].cuda()
            clean_image = data[1].numpy()

            restored_image = model(udc_image)
            restored_image = restored_image.cpu().detach()
            restored_image = (np.transpose(np.array(restored_image)[0], (1, 2, 0)) * 1)
            restored_image = np.clip(restored_image, 0, 1)

            # clean_image = clean_image.cpu().detach()
            clean_image = (np.transpose(np.array(clean_image)[0], (1, 2, 0)) * 1)

            # cv2.imwrite('./result/' + str(idx).zfill(3) + '.png',
            #            cv2.cvtColor((restored_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            test_psnr.append(PSNR(clean_image, restored_image))
            test_ssim.append(SSIM(clean_image, restored_image))
            print(str(idx) + ' : ' + str(PSNR(clean_image, restored_image)))

        psnr = sum(test_psnr) / len(test_psnr)
        ssim = sum(test_ssim) / len(test_ssim)

        print('### TEST PSNR Average ###')
        print(psnr)
        print('### ### ### ### ###')
        print('### TEST SSIM Average ###')
        print(ssim)
        print('### ### ### ### ###')
        print()
        print()


if __name__ == '__main__':
    print('===== Test ===== ')
    test(test_data_path=data_path, model_type=opt.model_type, panel_type=opt.panel_type)