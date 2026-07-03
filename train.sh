GPUS=${@:1}
PY_ARGS=${@:2}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS python train_val.py ${PY_ARGS} --seed 444