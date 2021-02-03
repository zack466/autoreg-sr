# python train.py --model PartialSR --loss VGG16Partial --ckpt_name PartialSR_64_long --lr_size 64 --batch_size 64 --num_patches 4 --epochs 10 --metrics PSNR SSIM sample lr consistency

# python train.py --model PartialSR_3k --loss VGG16Partial --ckpt_name PartialSR_32_3k --lr_size 32 --batch_size 64 --num_patches 32 --epochs 20 --metrics PSNR SSIM sample lr consistency

# python train.py --model PartialSR --loss VGG16Partial --ckpt_name PartialSR_32_final --lr_size 32 --batch_size 64 --num_patches 32 --epochs 40 --metrics PSNR SSIM consistency sample lr
# python train.py --model ConvSR --loss VGG16Partial --ckpt_name ConvSR_32_final --lr_size 32 --batch_size 64 --num_patches 32 --epochs 20 --metrics PSNR SSIM consistency sample lr --pre_upscale True

python inference.py --model PartialSR --ckpt_name PartialSR_32_final --lr_size 32 --batch_size 64 --num_patches 32 --img_path ./0002x4.png --patches True
python inference.py --model ConvSR --ckpt_name ConvSR_32_final --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --separate True
python inference.py --model SRCNN --ckpt_name SRCNN_final --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --separate True

python train.py --model SRCNN --loss psnr_loss --ckpt_name SRCNN_classic --lr_size 16 --batch_size 32 --num_patches 4 --epochs 1 --metrics PSNR SSIM lr sample VGG16Partial --pre_upscale True

# python train.py --model PartialPixelGen --loss VGG16Partial --ckpt_name PartialPixelGen --lr_size 32 --batch_size 64 --num_patches 32 --epochs 20 --metrics PSNR SSIM consistency sample lr

# python inference.py --model PartialSR --ckpt_name PartialSR_32_long --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

# python inference.py --model PartialSR --ckpt_name PartialSR_32_long_BN --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

# python inference.py --model ConvSR --ckpt_name ConvSR_32_long_BN --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png

# python inference.py --model ConvSR --ckpt_name ConvSR_32_long --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png

# python inference.py --model PartialSR --ckpt_name PartialSR_32_long_l2 --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

# python inference.py --model PartialSR --ckpt_name PartialSR_32_long_BN_l2 --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

# python inference.py --model ConvSR --ckpt_name ConvSR_32_long_BN_l2 --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png

# python inference.py --model ConvSR --ckpt_name ConvSR_32_long_l2 --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png
