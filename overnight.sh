python inference.py --model PartialSR --ckpt_name PartialSR_32_long --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

python inference.py --model PartialSR --ckpt_name PartialSR_32_long_BN --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

python inference.py --model ConvSR --ckpt_name ConvSR_32_long_BN --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png

python inference.py --model ConvSR --ckpt_name ConvSR_32_long --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png

python inference.py --model PartialSR --ckpt_name PartialSR_32_long_l2 --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

python inference.py --model PartialSR --ckpt_name PartialSR_32_long_BN_l2 --lr_size 32 --batch_size 64 --num_patches 32 --patches True --img_path ./0002x4.png

python inference.py --model ConvSR --ckpt_name ConvSR_32_long_BN_l2 --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png

python inference.py --model ConvSR --ckpt_name ConvSR_32_long_l2 --lr_size 32 --batch_size 64 --num_patches 32 --pre_upscale True --patches True --img_path ./0002x4.png
