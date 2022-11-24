import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import argparse
import pdb


def main():
    parser = argparse.ArgumentParser(description="testing mse")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        img_size = 32
    elif args.dataset == 'celeba':
        img_size = 64

    transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])

    loss = torch.nn.MSELoss()

    input1_path = './images/mickey.png'
    input1_pil = Image.open(input1_path)
    input1_t = transform(input1_pil)

    input2_names = os.listdir(args.data_dir)
    sum, count = 0, 0
    with torch.no_grad():
        for i in range(len(input2_names)):
            input2_path = os.path.join(args.data_dir, input2_names[i])
            input2_pil = Image.open(input2_path)
            input2_t = transform(input2_pil)

            sum += loss(input1_t, input2_t)
            count += 1
        print((sum/count).item())


if __name__ == "__main__":
    main()