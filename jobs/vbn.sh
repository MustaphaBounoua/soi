#!/bin/bash
source myvenv/bin/activate

i=0

for seed in   23 15  50 77 33
do
   for setting in  0  1
   do
         for dim in   10 25 50 
         do
            i=0
            for time_bin in  0 1 2 3 4 
            do
               export CUDA_VISIBLE_DEVICES="$i"  
               echo "GPU $i:"
               i=$((i+1))
               i=$((i%4))
               
               python -m experiments.run_soi_vbn --time_bin $time_bin --arch "mlp"  --setting $setting --seed $seed --change "change" --max_epoch 400 --lr 0.01 --bs 128 --dim $dim --importance_sampling  --use_ema --weight_s_functions --results_dir "out_vbn_final" &
               pids[${i}]=$!
            done
         
            for time_bin in  0  1 2 3 4 
            do
               export CUDA_VISIBLE_DEVICES="$i"  
               echo "GPU $i:"
               i=$((i+1))
               i=$((i%4))
               
               python -m experiments.run_soi_vbn --setting $setting --time_bin $time_bin --seed $seed --change "non_change" --max_epoch 200 --lr 0.01 --bs 128 --dim $dim --importance_sampling --use_ema  --weight_s_functions --results_dir "out_vbn_final" &
               pids[${i}]=$!
            done

            for pid in ${pids[*]}; do
               wait $pid
            done

         done
         wait -n
   done
done