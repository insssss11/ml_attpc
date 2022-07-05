firstDepths=("256")
dnnUnits=("256")
kernels=("5")
for firstDepth in "${firstDepths[@]}"
do
	for kernel in "${kernels[@]}"
	do
		for dnnUnit in "${dnnUnits[@]}"
		do
			echo "----------------------------------------------------------------------------------------------------------------"
			python3 ${ML_ATTPC}_mk2/tune_cnn_reg.py $1 400 4 9 -f ${firstDepth} -c ${kernel} -d ${dnnUnit} -o Ek -n Ek
			python3 ${ML_ATTPC}_mk2/tune_cnn_reg.py $1 400 4 9 -f ${firstDepth} -c ${kernel} -d ${dnnUnit} -o x y -n xy --dl
			python3 ${ML_ATTPC}_mk2/tune_cnn_reg.py $1 400 4 9 -f ${firstDepth} -c ${kernel} -d ${dnnUnit} -o trkLen -n L
			python3 ${ML_ATTPC}_mk2/tune_cnn_reg.py $1 400 4 9 -f ${firstDepth} -c ${kernel} -d ${dnnUnit} -o z -n z
		done
	done
done
