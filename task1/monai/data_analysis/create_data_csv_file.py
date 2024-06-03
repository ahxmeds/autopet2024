#%%
import os 
import pandas as pd 
import SimpleITK as sitk 
from glob import glob  
import numpy as np 
import sys 
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
from metrics.metrics import (
    get_3darray_from_niftipath
)
# %%
datadir = '/data/blobfuse/default/autopet2024/data/'
ctpaths = sorted(glob(os.path.join(datadir, 'imagesTr', '*0000.nii.gz')))
ptpaths = sorted(glob(os.path.join(datadir, 'imagesTr', '*0001.nii.gz')))
gtpaths = sorted(glob(os.path.join(datadir, 'labelsTr', '*.nii.gz')))

fdg_metadata = pd.read_csv('/data/blobfuse/default/autopet2024/data/fdg_metadata.csv')
psma_metadata = pd.read_csv('/data/blobfuse/default/autopet2024/data/psma_metadata.csv')
# %%
PATIENTID, TRACER, DIAGNOSIS, AGE, SEX = [],[],[],[],[]
SzX, SzY, SzZ = [],[],[]
SpX, SpY, SpZ = [],[],[]

for index, (ctpath, ptpath, gtpath) in enumerate(zip(ctpaths, ptpaths, gtpaths)):
    patientid = os.path.basename(gtpath)[:-7]
    PATIENTID.append(patientid) 
    if patientid.startswith('fdg'):
        TRACER.append('FDG')
        patientid_mutate = 'PETCT_' + patientid.split('_')[1]
        patientinfo = fdg_metadata[fdg_metadata['Subject ID'] == patientid_mutate]
        DIAGNOSIS.append(patientinfo.iloc[0]['diagnosis'])
        AGE.append(int(patientinfo.iloc[0]['age'][:-1]))
        SEX.append(patientinfo.iloc[0]['sex'])

    else: 
        patientid_mutate = 'PSMA_' + patientid[5:-11]
        studydate = patientid[-10:]
        patientinfo = psma_metadata[psma_metadata['Subject ID'] == patientid_mutate]
        patientinfo = patientinfo[patientinfo['Study Date'] == studydate]
        TRACER.append(patientinfo['pet_radionuclide'].item())
        gt = get_3darray_from_niftipath(gtpath)
        if np.all(gt == 0):
            DIAGNOSIS.append('NEGATIVE')
        else:
            DIAGNOSIS.append('PROSTATE')
        AGE.append(patientinfo['age'].item())
        SEX.append('M')
    
    gtimg = sitk.ReadImage(gtpath)
    SzX.append(gtimg.GetSize()[0])
    SzY.append(gtimg.GetSize()[1])
    SzZ.append(gtimg.GetSize()[2])
    SpX.append(gtimg.GetSpacing()[0])
    SpY.append(gtimg.GetSpacing()[1])
    SpZ.append(gtimg.GetSpacing()[2])
    print(f'{index}: Done with patientID = {patientid}')


# %%
col_names = [
    'PatientID', 'Tracer', 'Diagnosis','Age', 'Sex',
    'SizeX','SizeY','SizeZ','SpcX','SpcY','SpcZ'
]
data = [PATIENTID, TRACER, DIAGNOSIS, AGE, SEX, SzX, SzY, SzZ, SpX, SpY, SpZ]
datainfo = pd.DataFrame(columns=col_names)
for index, col in enumerate(col_names):
    datainfo[col] = data[index]
datainfo.to_csv('datainfo.csv', index=False)

#%%
# datainfo = pd.read_csv('/home/jhubadmin/Projects/autopet2024/task1/monai/data_analysis/datainfo.csv')
# TRACER = datainfo['Tracer'].tolist()
# #%%
# for index, t in enumerate(TRACER):
#     if t == '18F':
#         TRACER[index] = '18F-PSMA'
#     elif t == '68Ga':
#         TRACER[index] = '68Ga-PSMA'
#     else:
#         pass

# # %%
# datainfo['Tracer'] = TRACER
# datainfo.to_csv('datainfo.csv', index=False)

# %%
