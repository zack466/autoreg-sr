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
import os
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--up_factor", type=int, default=4)
    parser.add_argument("--lr_size", type=int, default=16)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--show_example", type=bool, default=False)
    parser.add_argument("--show_summary", type=bool, default=False)
    parser.add_argument("--show_results", type=bool, default=False)
    parser.add_argument("--ckpt_every", type=int, default=-1)
    parser.add_argument("--ckpt_name", type=str, default="")
    parser.add_argument("--metrics", nargs="+")
    return parser.parse_args()


def main():
    config = parse_args()
    using_mask = config.model in ["PConvSR", "PConvResNet"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    div2k = datasets.Div2K(
        size=config.lr_size,
        factor=config.up_factor,
        mult=config.num_patches,
        cache_size=config.cache_size,
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

    training_dataloader = torch.utils.data.DataLoader(
        training_set, batch_size=config.batch_size
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_set, batch_size=config.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=config.batch_size
    )

    # model
    try:
        model = getattr(models, config.model)().to(device)
    except:
        print(f"Model {config.model} not found. Quitting...")
        sys.exit(1)

    # optimizer
    lr = config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=50
    # )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5)

    # loss
    if config.loss == "L1":
        loss_func = torch.nn.L1Loss()
    elif config.loss == "L2" or config.loss == "MSE":
        loss_func = torch.nn.MSELoss()
    elif config.loss == "VGG19":
        loss_func = losses.VGG16PartialLoss().to(device)
    else:
        try:
            loss_func = getattr(losses, config.loss)
        except:
            print(f"loss {config.loss} not found.")
            sys.exit(1)

    if config.show_summary:
        input_size = (3, hr_size, hr_size)
        if using_mask:
            input_size = [input_size, input_size]
        torchsummary.summary(model, input_size=input_size)

    writer = SummaryWriter(f"./models/outputs/{config.ckpt_name}")

    try:  # hacky workaround, doesnt load checkpoint if doesnt exist
        assert config.ckpt_name
        checkpoint = torch.load(
            f"./models/outputs/{config.ckpt_name}/" + config.ckpt_name + ".pt"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        training_loss = checkpoint["training_loss"]
        validation_loss = checkpoint["validation_loss"]
        print("Loading from previous checkpoint")
    except:
        start_epoch = 0
        training_loss = AverageMeter("Training")
        validation_loss = AverageMeter("Validation")
        print("Not loading from checkpoint")

    def loop(dataloader, epoch, loss_meter, back=True):
        for i, batch in enumerate(dataloader):
            step = epoch * len(dataloader) + i

            if back:
                optimizer.zero_grad()

            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)

            if using_mask:
                with torch.no_grad():
                    upscaled, mask_in = image_mask(lr, config.up_factor)
                pred, mask_out = model(upscaled.to(device), mask_in.to(device))
            else:
                with torch.no_grad():
                    upscaled = transforms.functional.resize(lr, (hr_size, hr_size))
                pred = model(upscaled)

            if config.loss == "VGG19":
                loss, _, _ = loss_func(pred, hr)  # VGG style loss
            else:
                loss = loss_func(pred, hr)

            if back:
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), writer, step)

            with torch.no_grad():
                for metric in config.metrics:
                    tag = loss_meter.name + "/" + metric
                    if metric == "PSNR":
                        writer.add_scalar(tag, losses.psnr(pred, hr), step)
                    elif metric == "SSIM":
                        writer.add_scalar(tag, losses.ssim(pred, hr), step)
                    elif metric == "consistency":
                        downscaled_pred = transforms.functional.resize(
                            pred, (config.lr_size, config.lr_size)
                        )
                        writer.add_scalar(
                            tag,
                            torch.nn.functional.mse_loss(downscaled_pred, lr).item(),
                            step,
                        )
                    elif metric == "lr":
                        writer.add_scalar(tag, lr_scheduler.get_last_lr()[0], step)
        lr_scheduler.step()

    for epoch in range(start_epoch, start_epoch + config.epochs):
        loop(training_dataloader, epoch, training_loss)
        print(f"Epoch {epoch}: {training_loss}")
        with torch.no_grad():
            loop(validation_dataloader, epoch, validation_loss, back=False)
            print(f"Epoch {epoch}: {validation_loss}")

        if config.ckpt_every != -1 and epoch % config.ckpt_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss.val,
                },
                f"./models/outputs/{config.ckpt_name}/"
                + config.ckpt_name
                + f"_{epoch}.pt",
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss.val,
                },
                f"./models/outputs/{config.ckpt_name}/" + config.ckpt_name + ".pt",
            )

    testing_loss = AverageMeter("Testing")
    loop(test_dataloader, 0, testing_loss)
    print(f"Test: {testing_loss}")

    if config.ckpt_name:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            },
            f"./models/outputs/{config.ckpt_name}/" + config.ckpt_name + ".pt",
        )


if __name__ == "__main__":
    main()