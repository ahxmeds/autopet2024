#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os 

AUTOPET_SEGMENTATION_FOLDER = '/data/blobfuse/default/autopet2024' # path to the directory containing `data` and `results` (this will be created by the pipeline) folders.

DATA_FOLDER = os.path.join(AUTOPET_SEGMENTATION_FOLDER, 'data')
RESULTS_FOLDER = os.path.join('/data/blobfuse/autopet2024', 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
WORKING_FOLDER = os.path.dirname(os.path.abspath(__file__))
