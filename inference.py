import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchsummary
import utils
from utils import AverageMeter, image_mask
import models
import datasets
import losses
import argparse
import sys
import numpy as np


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
    return parser.parse_args()


def main():
    config = parse_args()
    using_mask = config.model in ["PConvSR", "PConvResNet"]

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
    try:
        model = getattr(models, config.model)().to(device)
    except:
        print(f"Model {config.model} not found.")
        sys.exit(1)

    if config.show_summary:
        input_size = (3, hr_size, hr_size)
        if using_mask:
            input_size = [input_size, input_size]
        torchsummary.summary(model, input_size=input_size)

    try:
        assert config.ckpt_name
        checkpoint = torch.load(
            f"./models/outputs/{config.ckpt_name}/" + config.ckpt_name + ".pt"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        training_loss = checkpoint["training_loss"]
        validation_loss = checkpoint["validation_loss"]
        print("Loading from previous checkpoint")
    except:
        print("Must load a model from a checkpoint or error loading model")
        sys.exit(1)

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

    for i in range(32, 48):
        with torch.no_grad():
            lr, hr = test_set[i]
            batch = utils.example_batch(torch.tensor(lr).to(device), config.batch_size)
            if using_mask:
                upscaled, mask_in = image_mask(batch, 4)
                pred, _ = model(upscaled.to(device), mask_in.to(device))
            else:
                upscaled = torchvision.transforms.functional.resize(
                    torch.tensor(batch), (hr_size, hr_size)
                )
                pred = model(upscaled)
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


if __name__ == "__main__":
    main()