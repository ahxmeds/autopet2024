# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

torchrun --standalone --nproc_per_node=1 trainddp.py --fold=1 --network-name='unet' --epochs=4 --input-patch-size=192 --train-bs=1 --num_workers=2 --lr=2e-4 --wd=1e-5 --val-interval=2 --sw-bs=2 --cache-rate=1