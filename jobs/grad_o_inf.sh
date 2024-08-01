


i=0
for dim in     10 20 # 15 5
do
   i=0 
   k=0
   for setting in 0  1 
   do
      for seed in 77 55 33 #22 #11 
      do 
            export CUDA_VISIBLE_DEVICES="$k"  
            echo "GPU $k:"
            i=$((i+1))
            k=$((i%4))
            python -m experiments.run_soi_grad --benchmark "mix" --arch "mlp" --rho 0.7 --seed $seed --o_inf_order 2 --setting $setting --dim $dim --bs 256 --lr 0.01  --max_epoch 500 --use_ema  --test_epoch 250 --results_dir "results/paper/" & 
            pids[${i}]=$!
      done
      echo "Did nb_mod : $nb_mod:"
   done
   for pid in ${pids[*]}; do
            wait $pid
   done
   wait -n
   echo "Did dim : $dim:"
done