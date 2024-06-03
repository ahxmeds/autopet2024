#%%
import os 
from glob import glob 
# %%
datadir = '/data/blobfuse/default/autopet2024/data/'
ctpaths = sorted(glob(os.path.join(datadir, 'imagesTr', '*0000.nii.gz')))
ptpaths = sorted(glob(os.path.join(datadir, 'imagesTr', '*0001.nii.gz')))
gtpaths = sorted(glob(os.path.join(datadir, 'labelsTr', '*.nii.gz')))
# %%
count = 0
for ctpath, ptpath, gtpath in zip(ctpaths, ptpaths, gtpaths):
    ctfname = os.path.basename(ctpath)
    if ctfname.startswith('fdg'):
        ctpath_new = os.path.join(os.path.dirname(ctpath), f'{os.path.basename(ctpath)[:25]}_0000.nii.gz')
        ptpath_new = os.path.join(os.path.dirname(ptpath), f'{os.path.basename(ptpath)[:25]}_0001.nii.gz')
        gtpath_new = os.path.join(os.path.dirname(gtpath), f'{os.path.basename(gtpath)[:25]}.nii.gz')

        os.rename(ctpath, ctpath_new)
        os.rename(ptpath, ptpath_new)
        os.rename(gtpath, gtpath_new)
        print(count)
        count += 1
    else:
        pass
# %%
