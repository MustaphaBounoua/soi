#!/bin/bash
source myvenv/bin/activate


i=0
for dim in  5  15  10 20 
do
   for setting in 0  1 
   do
      for seed in 77 55  33 22 11
      do
         i=0
         for rho in   0.7 0.65 0.6 0.55 0.5 0.475 0.45 0.425 0.4 0.375 0.35 0.325 0.3 0.25 0.2 0.1
         #for rho in   0.7 0.65  0.6  0.5  0.45  0.4 0.375 0.35 0.3 0.25 0.2 0.1
         do
            export CUDA_VISIBLE_DEVICES="$i"  
            i=$((i+1))
            i=$((i%4))
            echo "GPU $i:"
            python -m experiments.run_soi --benchmark $1 --arch "mlp" --rho $rho --seed $seed --setting $setting --dim $dim --bs 256 --lr 0.01 --max_epoch 500  --use_ema --importance_sampling  --weight_s_functions & 
            pids[${i}]=$!
         done

         for pid in ${pids[*]}; do
            wait $pid
         done
         echo "Did seed $seed:"
      done
      wait -n
      echo "Did setting : $setting:"
   done
   wait -n
   echo "Did dim : $dim:"
done







