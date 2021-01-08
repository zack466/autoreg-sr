import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchsummary
from torchvision.transforms.transforms import PILToTensor
import utils
from utils import AverageMeter, image_mask
import models
import datasets
import losses
import argparse
import sys
import numpy as np
import PIL
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--up_factor", type=int, default=4)
    parser.add_argument("--lr_size", type=int, default=16)
    parser.add_argument("--num_patches", type=int, default=4)
    parser.add_argument("--cache_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--show_example", type=bool, default=False)
    parser.add_argument("--show_summary", type=bool, default=False)
    parser.add_argument("--show_results", type=bool, default=False)
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--unsupervised", type=bool, default=False)
    parser.add_argument("--patches", type=bool, default=False)
    parser.add_argument("--pre_upscale", type=bool, default=False)
    return parser.parse_args()


def main():
    config = parse_args()
    using_mask = config.model in ["PConvSR", "PConvResNet"] or "Partial" in config.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    div2k = datasets.Div2K(
        config.lr_size, config.up_factor, config.num_patches, config.cache_size
    )

    d = config.num_patches
    training_set, validation_set, test_set = torch.utils.data.random_split(
        div2k, [640 * d, 80 * d, 80 * d]
    )
    hr_size = config.lr_size * config.up_factor

    if config.show_example:
        lr, hr = div2k[6]
        print(lr.shape)
        print(hr.shape)
        utils.showImages([lr, hr])

    # model
    if config.model == "PixelCNN":
        model = models.PixelCNN(3, 3).to(device)
    elif config.model == "PartialSR" or config.model == "ConvSR":
        if "BN" in config.ckpt_name:
            model = getattr(models, config.model)().to(device)
        else:
            model = getattr(models, config.model)(batch_norm=False).to(device)
    else:
        try:
            model = getattr(models, config.model)().to(device)
        except:
            print(f"Model {config.model} not found. Quitting...")
            sys.exit(1)

    if config.show_summary:
        input_size = (3, hr_size, hr_size)
        if using_mask:
            input_size = [input_size, input_size]
        torchsummary.summary(model, input_size=input_size)

    # for pretrained srresnet
    # weights = torch.load(
    #     f"./models/outputs/{config.ckpt_name}/" + config.ckpt_name + ".pt"
    # )
    # model = weights["model"]

    # try:
    assert config.ckpt_name
    checkpoint = torch.load(
        f"./models/outputs/{config.ckpt_name}/" + config.ckpt_name + ".pt"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    # start_epoch = checkpoint["epoch"]
    # training_loss = checkpoint["training_loss"]
    # validation_loss = checkpoint["validation_loss"]
    print("Loading from previous checkpoint")
    # except:
    #     print("Must load a model from a checkpoint or error loading model")
    #     sys.exit(1)

    # for i, batch in enumerate(dataloader):

    #     lr, hr = batch
    #     lr, hr = lr.to(device), hr.to(device)

    #     if using_mask:
    #         with torch.no_grad():
    #             upscaled, mask_in = image_mask(lr, config.up_factor)
    #         pred, mask_out = model(upscaled.to(device), mask_in.to(device))
    #     else:
    #         with torch.no_grad():
    #             upscaled = transforms.functional.resize(lr, (hr_size, hr_size))
    #         pred = model(upscaled)

    #     if config.loss == "VGG19":
    #         loss, _, _ = loss_func(pred, hr)  # VGG style loss
    #     else:
    #         loss = loss_func(pred, hr)

    #     break

    if config.img_path:
        try:
            img = Image.open(config.img_path)
        except:
            print(f"Image {config.img_path} not found")
            sys.exit(1)
        with torch.no_grad():
            img = transforms.ToTensor()(img)
            c, height, width = img.shape
            s = config.lr_size
            rows = []
            for h in range(height // s):
                row = []
                for w in range(width // s):
                    patch = transforms.functional.crop(img, h * s, w * s, s, s).to(
                        device
                    )
                    batch = utils.example_batch(patch, config.batch_size)
                    if using_mask:
                        upscaled, mask_in = image_mask(batch, config.up_factor)
                        pred = model(upscaled.to(device), mask_in.to(device))
                    elif config.pre_upscale:
                        upscaled = torchvision.transforms.functional.resize(
                            torch.tensor(batch), (hr_size, hr_size)
                        )
                        pred = model(upscaled)
                    else:
                        pred = model(batch)
                    out = pred.cpu()[0]  # 3, 64, 64
                    row.append(out)
                full_row = torch.cat(row, -1)  # 3, 64, 1984
                rows.append(full_row)
            img_out = torch.cat(rows, 1).permute(1, 2, 0)
            img_out = np.uint8(np.array(img_out * 255))
            img_out = Image.fromarray(img_out)
            img_out.save(f"./models/outputs/{config.ckpt_name}/out.png")

    if config.patches:
        for i in range(0, 48):
            with torch.no_grad():
                lr, hr = test_set[i]
                batch = utils.example_batch(
                    torch.tensor(lr).to(device), config.batch_size
                )
                if using_mask:
                    upscaled, mask_in = image_mask(batch, config.up_factor)
                    pred = model(upscaled.to(device), mask_in.to(device))
                elif config.pre_upscale:
                    upscaled = torchvision.transforms.functional.resize(
                        torch.tensor(batch), (hr_size, hr_size)
                    )
                    pred = model(upscaled)
                else:
                    upscaled = torchvision.transforms.functional.resize(
                        torch.tensor(batch), (hr_size, hr_size)
                    )
                    pred = model(batch)
                bicubic = torchvision.transforms.functional.resize(
                    torch.tensor(lr), (hr_size, hr_size)
                )
                utils.saveImages(
                    [
                        np.asarray(lr),
                        np.array(upscaled[0].cpu()),
                        np.array(bicubic),
                        np.array(pred[0].cpu()),
                        hr,
                    ],
                    f"./models/outputs/{config.ckpt_name}/output_{i}.png",
                )
        print(f"Outputs saved to ./models/outputs/{config.ckpt_name}/")

    if config.unsupervised:
        with torch.no_grad():
            model(
                torch.ones(config.batch_size, 3, config.lr_size, config.lr_size).to(
                    device
                )
            )
        num_imgs = 16
        imgs = model.sample(n_samples=num_imgs)
        for i in range(num_imgs):
            img = imgs[i]
            img = (img * 255).permute(1, 2, 0).cpu()
            img = np.uint8(np.array(img))
            img_out = Image.fromarray(img)
            img_out.save(f"./models/outputs/{config.ckpt_name}/sample{i}.png")


if __name__ == "__main__":
    main()
