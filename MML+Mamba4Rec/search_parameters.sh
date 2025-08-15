#!/bin/bash
for ll_loss_weight in 1.0
do
  for lg_loss_weight in 0.1 0.5 1.0 1.5 2.0
  do
    for gg_loss_weight in 1.0
    do
      for behavior_regularizer_weight in 0.001
      do
        for argumentation_ratio in 0.1 0.3 0.5 0.7 0.9
        do
        for cls_weight in 0.001
        do
          for temperature in 0.05 0.1 0.2 0.5 1.0
            do
              CUDA_VISIBLE_DEVICES=0, python main.py --train_dir=default --maxlen=300 --batch_size=128 --num_epochs=1000 --device=cuda --hidden_units=50 --interval=20 --load_processed_data=true --l2_emb=0.001 --behavior_regularizer_weight=${behavior_regularizer_weight} --ll_loss_weight=${ll_loss_weight} --lg_loss_weight=${lg_loss_weight} --gg_loss_weight=${gg_loss_weight} --num_domain_shared_blocks=0 --num_domain_specific_blocks=2 --num_blocks_mixed_seq=2  --temperature=${temperature} --cls_weight=${cls_weight}  --argumentation_methods 'crop' --notes 'MML_Mamba4Rec_search_parameters' --argumentation_ratio=${argumentation_ratio}
              done
          done
        done
      done
    done
  done
done