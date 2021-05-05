python get_snap_names.py
level=$1
model=$2

if [ $2 -z ]
then
model="cResNet39"
fi

if [ $1 -z ]
then
level=1
fi

while IFS= read -r ckpt
do
python c_evaluate.py --model $model --level $level --restore-from $ckpt --include-hyper
done < tmp

rm tmp
