img=$1

python c_inference.py $img ../results/c42_try1/cResNet42-lvl1.ckpt-180000 --save-original --data-path dataset/VOC2012/ --auto --no-gpu

python c_inference.py $img ../results/sigma_resblock2/cResNet-sigma-resblock-lvl1.ckpt-180000 --save-original --data-path dataset/VOC2012/ --auto --no-gpu

python inference.py $img deeplab_resnet.ckpt --level 1 --data-path dataset/VOC2012/JPEGImages/ --save-original --no-gpu

python inference.py $img deeplab_resnet.ckpt --data-path dataset/VOC2012/JPEGImages/ --save-original --no-gpu
