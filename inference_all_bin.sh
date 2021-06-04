img=$1

python c_inference.py $img ../results/binary/resblock-finetune/cResNet40-h3-lvl1-bin.ckpt-14000 --with-original --data-path dataset/VOC2012/ --auto --no-gpu

python c_inference.py $img ../results/binary/c42-finetune/cResNet42-lvl1-bin.ckpt-9000 --with-original --data-path dataset/VOC2012/ --auto --no-gpu

python inference.py $img ../results/binary/anchors/model-bin.ckpt-20000 --level 1 --data-path dataset/VOC2012/JPEGImages/ --no-gpu --num-classes 2

python inference.py $img ../results/binary/anchors/model-bin.ckpt-20000 --data-path dataset/VOC2012/JPEGImages/ --no-gpu --num-classes 2
