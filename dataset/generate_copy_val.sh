cp val.txt val-lvl$1.txt
sed -i 's/.jpg/.png/g' val-lvl$1.txt
cmd=$(printf 's/JPEGImages/Compressed\/level%s-val/g' $1)
sed -i $cmd val-lvl$1.txt
