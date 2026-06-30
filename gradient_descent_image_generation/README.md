# pRF MEI Optimization

This directory is independent from `/home/hanfeig/MEI`. That source repository
is treated as a read-only dependency and is only used to load `prf_utils.py`.

## Direct run

```bash
python maximize_prf_activation.py \
  --source_repo /home/hanfeig/MEI \
  --model_name CLIP_RN50 \
  --layer_name layer1 \
  --best_prf_idx_path /path/to/layer1/best_prf_idx.npy \
  --voxel_indices 0 \
  --save_dir /path/to/output \
  --device cuda
```

## Slurm array

Edit the paths in `scripts/maximize_prf_activation_tasks.example.json` and
`scripts/maximize_prf_activation_array.job`, then submit:

```bash
sbatch scripts/maximize_prf_activation_array.job
```

## S1 ADV_RN50 set1: 50 V1 images

`set1` and `set2` are complementary 50/50 partitions of the original train,
validation, and nested-validation image sets. They are not different random
initializations of the ADV backbone.

The dedicated array job selects the 50 S1 V1 voxels with the highest NSD noise
ceiling. The fixed voxel list and its selection metadata are stored in
`scripts/adv_v1_top50_noise_ceiling.json`. It uses `layer4`, 100 optimization
steps, a 30-minute task limit, and at most two concurrent GPUs by default.

Install the environment once:

```bash
bash scripts/setup_mei_environment.sh
```

Submit one voxel first:

```bash
sbatch --array=0 scripts/maximize_adv_v1_50_array.job
```

After that job succeeds, submit all 50 voxels, with at most two running at
once:

```bash
sbatch scripts/maximize_adv_v1_50_array.job
```

The default output directory is:

```text
/user_data/hanfeig/prf_max_activation/split_1_zscore_drop/S1/ADV_RN50_set1/layer4/V1_top50_noise_ceiling
```

The layer and other settings can be overridden without editing the job file:

```bash
sbatch --export=ALL,LAYER_NAME=layer3,STEPS=100 scripts/maximize_adv_v1_50_array.job
```
