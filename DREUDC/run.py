import cv2
import numpy as np
import torch
import argparse
from metrics import *
from encoder import Encoder
from models import Former_Freq_Embed
from torch.utils.data import DataLoader
from data import ImageDataset_raw2rgb

device = torch.device('cuda')

encoder_path = './pretrained_models/dreudc/encoder.pth'
axon_path = './pretrained_models/dreudc/axon.pth'
zfold_path = './pretrained_models/dreudc/zfold.pth'
data_path = './dataset/'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--panel_type', type=str, default='axon', help='type of dataset')

opt = parser.parse_args()


def load_models(enc_load_path, panel_type):
    enc = Encoder()
    enc.to(device)
    checkpoint = torch.load(enc_load_path, map_location='cuda')
    enc.load_state_dict(checkpoint)

    model = Former_Freq_Embed(channel=32)
    model.to(device)

    if panel_type == 'axon':
        checkpoint = torch.load(axon_path, map_location='cuda')
        model.load_state_dict(checkpoint)

    elif panel_type == 'zfold':
        checkpoint = torch.load(zfold_path, map_location='cuda')
        model.load_state_dict(checkpoint)

    else:
        print('Unidentified panel type, please check.')

    for param in enc.parameters():
        param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False

    enc.eval()
    model.eval()

    return enc, model


### initialize test & performance metrics ###

test_psnr = []
test_ssim = []


def test(enc_load_path, test_data_path, panel_type):
    # load dataset
    test_dataset = ImageDataset_raw2rgb(image_dirs=test_data_path, panel_type=panel_type)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # load model
    enc, model = load_models(enc_load_path=enc_load_path, panel_type=panel_type)

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            udc_image = data[0].cuda()
            clean_image = data[1].numpy()

            representation, _ = enc(udc_image)
            restored_image = model(udc_image, representation)
            restored_image = restored_image.cpu().detach()
            restored_image = (np.transpose(np.array(restored_image)[0], (1, 2, 0)) * 1)
            restored_image = np.clip(restored_image, 0, 1)

            # clean_image = clean_image.cpu().detach()
            clean_image = (np.transpose(np.array(clean_image)[0], (1, 2, 0)) * 1)

            cv2.imwrite('./result/' + str(idx).zfill(3) + '.png',
                        cv2.cvtColor((restored_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
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
    test(enc_load_path=encoder_path, test_data_path=data_path, panel_type=opt.panel_type)
