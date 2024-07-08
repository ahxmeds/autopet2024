# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
python inference.py --fold=0 --network-name='unet' --input-patch-size=128 --inference-patch-size=192 --num_workers=4  --sw-bs=2 --val-interval=4 