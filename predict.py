import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image

from unet import UNet
from dataset import BasicDataset
from conv2d3d import Conv2d3d

from scipy.io import loadmat


def predict_img(net,
                full_img,
                n_classes,
                device,
                scale_factor,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor)).type(torch.FloatTensor)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    w, h = full_img.shape

    with torch.no_grad():
        
        output = net(img)
        
        full_mask = np.arange(w * h * n_classes).reshape((w, h, n_classes)).astype(float)
        
        for i in range(n_classes):
            full_mask[:,:,i] = np.array(Image.fromarray(output[:,i,:,:].squeeze().cpu().numpy()).resize((h, w))).astype(float)
            full_mask[:,:,i] = full_mask[:,:,i] - np.min(full_mask[:,:,i])
            full_mask[:,:,i] = full_mask[:,:,i] / np.max(full_mask[:,:,i])

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output
    return out_files


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=6)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
    
        logging.info("\nPredicting image {} ...".format(fn))
        img = loadmat(fn)['depth']
        mask = predict_img(net=net,
                           full_img=img,
                           n_classes=net.n_classes,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
                           
        if not args.no_save:
            Conv2d3d.plot(img, mask, 'results/' + fn[-11:-4] + '.png')
            logging.info("Mask saved to {}".format('results/' + fn[-11:-4] + '.png'))
