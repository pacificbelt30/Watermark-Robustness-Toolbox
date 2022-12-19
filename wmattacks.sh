base=$1
echo $base
arr=`ls $base`
echo $arr
arr=`ls $base | grep -v jia`
ary=(`echo $arr`)
for i in `seq 1 ${#ary[@]}`
do
	echo bash allattacks.sh $base${ary[$i-1]}
    bash allattacks.sh $base${ary[$i-1]}
done
