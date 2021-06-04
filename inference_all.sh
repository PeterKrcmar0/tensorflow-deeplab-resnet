img=$1

python c_inference.py $img ../results/c42_try1/cResNet42-lvl1.ckpt-180000 --with-original --data-path dataset/VOC2012/ --auto --no-gpu

python c_inference.py $img ../results/sigma_resblock2/cResNet40-h3-lvl1.ckpt-180000 --with-original --data-path dataset/VOC2012/ --auto --no-gpu

python inference.py $img deeplab_resnet.ckpt --level 1 --data-path dataset/VOC2012/JPEGImages/ --no-gpu

python inference.py $img deeplab_resnet.ckpt --data-path dataset/VOC2012/JPEGImages/ --no-gpu
