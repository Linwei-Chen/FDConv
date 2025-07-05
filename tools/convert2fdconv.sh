python ./convert_to_fdconv.py \
--model_type resnet \
--weight_path ./resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
--save_path ./resnet18_8xb32_in1k_20210831-fbbb1da6_FDConv.pth

python ./convert_to_fdconv.py \
--model_type vit \
--weight_path ./mmseg_deit_small_patch16_224-cd65a155.pth \
--save_path ./mmseg_deit_small_patch16_224-cd65a155_FDConv.pth