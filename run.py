!python main.py --task 1 --seeds 0 1 2 3 4 --drug TE --model densenet169   --threshold_method min_recall --threshold_min_recall 0.85   --best_metric recall_resistant 2>&1 | tee ./training_log.txt

python main.py --task 1 --seeds 0 1 2 3 4 --drug TE --model resnet50 densenet169 efficientnet_b3 convnext_tiny convnextv2_tiny  --threshold_method balanced   --best_metric f1_resistant   2>&1 | tee ./training_log_CIP.txt
python main.py --task 2 --seeds 0 1 2 3 4 --drug CIP   --model resnet50 densenet169 efficientnet_b3 convnext_tiny convnextv2_tiny   --threshold_method balanced   --best_metric f1_resistant   2>&1 | tee ./training_log_CIP_task2.txt


python main.py --task 1 --seeds 0 2 4 10 42 --drug CIP --model resnet50 densenet169 efficientnet_b3 convnext_tiny convnextv2_tiny  --threshold_method min_recall   --best_metric recall_resistant   2>&1 | tee ./training_log_CIP.txt


cd /home/son/new_mia/antibiotics_framework && conda run -n MIA python parse_log.py --from-results /home/son/new_mia/results --task 1 --drug CIP --out CIP_from_results.xlsx --both 2>&1

python main.py --task 1 --seeds 0 1 2 3 4 --drug CIP   --model mobilenetv2_100   --threshold_method min_recall --best_metric f1_resistant   --patience 10


# Task 1 multiclass
python main.py --task 1 --multiclass --model densenet121 --loss focal

# Task 2 multiclass (cần bước 1 xong trước)
python main.py --task 2 --multiclass --model densenet121 --loss focal


python main.py \
  --task 1 \
  --drug AZI \
  --model resnet50 densenet169 vgg19 efficientnet_b2 convnext_tiny convnextv2_tiny \
  --seeds 0 1 2 3 4 \
  --threshold_method pr_equal \
  --best_metric f1_resistant \
  --best_metric_level image_level \
  --loss focal \
  2>&1 | tee /home/aisia/new_mia/training_log_task1_prequal.txt &
echo "PID: $!"


cd antibiotics_framework
python export_patches.py --drug all --output_dir ../task2_patches


#Degrade 

cd /home/aisia/new_mia/antibiotics_framework

# CIP, model tốt nhất, tất cả degradation:
python degraded_mask_eval.py \
  --drug CIP \
  --model efficientnet_b5 \
  --seeds 0 1 2 3 4 \
  --out /home/aisia/new_mia/degraded_mask_CIP_effb5.xlsx

# Nhiều drugs + nhiều models:
python degraded_mask_eval.py \
  --drug CIP AZI \
  --model efficientnet_b5 vit_small convnext_base \
  --mode all \
  --bf16

# Chỉ test occlusion nhanh:
python degraded_mask_eval.py \
  --drug CIP \
  --model efficientnet_b5 \
  --mode occlusion \
  --seeds 0 1


  cd /home/aisia/new_mia/antibiotics_framework && python degraded_mask_eval.py   --drug CIP    --seeds 0 1 2 3 4   --mode erosion   --bf16   --results_dir /home/aisia/new_mia/results_erosion_aug   --data_root /home/aisia/new_mia --model convnextv2_tiny  --random_erosion_test --random_erosion_px 1 2 3   --random_erosion_p 0.5   --out /home/aisia/new_mia/degraded_mask_erosion_aug_effb5.xlsx 2>&1