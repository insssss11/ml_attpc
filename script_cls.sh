firstDepths=("64" "128")
dnnUnits=("64" "128")
kernels=("3" "5")
for firstDepth in "${firstDepths[@]}"
do
	for kernel in "${kernels[@]}"
	do
		for dnnUnit in "${dnnUnits[@]}"
		do
			echo "----------------------------------------------------------------------------------------------------------------"
			python3 ${ML_ATTPC}_mk2/tune_cnn_flg0.py $1 400 2 4 -f ${firstDepth} -c ${kernel} -d ${dnnUnit}
		done
	done
done
