cnt=1;
for file in `cat val.txt`;
do
let cnt=cnt+1;
if [ $cnt -eq 2 ];
then cp "../../datasets/voc2012$file" "./trainset/";
cnt=0;
fi;
done #mv "$file" /path/of/destination ; done
