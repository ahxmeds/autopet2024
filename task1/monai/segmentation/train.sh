# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

torchrun --standalone --nproc_per_node=4 trainddp.py --fold=1 --network-name='unet' --epochs=400 --input-patch-size=128 --inference-patch-size=192 --train-bs=4 --num_workers=16 --lr=2e-4 --wd=1e-5 --val-interval=4 --sw-bs=2 --cache-rate=0.02