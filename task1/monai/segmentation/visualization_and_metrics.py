#%%
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import SimpleITK as sitk 
import argparse
from initialize_train import (
    get_valid_pred_data_in_dict_format,
)
from metrics.metrics import (
    get_3darray_from_niftipath,
    get_voxel_spacing_from_niftipath,
    calculate_patient_level_dice_score,
    calculate_patient_level_false_positive_volume,
    calculate_patient_level_false_negative_volume
)
from joblib import Parallel, delayed
import os 
import sys 
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
#%%
def plot_and_save_coronal_sagittal_mip_visualization(
        ct, pt, gt, pr, 
        patientid, tracer, diagnosis, 
        dsc, fnv, fpv, 
        save_visual_dir
):
    fig, ax = plt.subplots(2, 4, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    
    ct_cor = np.rot90(np.max(ct, axis=1))
    pt_cor = np.rot90(np.max(pt, axis=1))
    gt_cor = np.rot90(np.max(gt, axis=1))
    pr_cor = np.rot90(np.max(pr, axis=1))

    ct_sag = np.fliplr(np.rot90(np.max(ct, axis=0)))
    pt_sag = np.fliplr(np.rot90(np.max(pt, axis=0)))
    gt_sag = np.fliplr(np.rot90(np.max(gt, axis=0)))
    pr_sag = np.fliplr(np.rot90(np.max(pr, axis=0)))

    ax[0][0].imshow(ct_cor)
    ax[0][1].imshow(pt_cor)
    ax[0][2].imshow(gt_cor)
    ax[0][3].imshow(pr_cor)

    ax[1][0].imshow(ct_sag)
    ax[1][1].imshow(pt_sag)
    ax[1][2].imshow(gt_sag)
    ax[1][3].imshow(pr_sag)

    ax[0][0].set_title('CT', fontsize=14)
    ax[0][1].set_title('PT', fontsize=14)
    ax[0][2].set_title('GT', fontsize=14)
    ax[0][3].set_title('Pred', fontsize=14)

    fig.suptitle(f'{patientid} | {tracer} | {diagnosis}\n DSC: {dsc:.4f}, FNV: {fnv:.4f}, FPV: {fpv:.4f}', fontsize=16)

    plt.subplots_adjust(hspace=0.07, wspace=0.05)
    fig.savefig(os.path.join(save_visual_dir, f'{patientid}.png'), dpi=200, bbox_inches='tight')
    plt.close('all')



def process_one_case(data, save_visual_dir):
    ctpath, ptpath, gtpath, prpath = data['CT'], data['PT'], data['GT'], data['PR']
    patientid = os.path.basename(gtpath)[:-7]
    spacing = get_voxel_spacing_from_niftipath(gtpath)
    datainfo = pd.read_csv('/home/jhubadmin/Projects/autopet2024/task1/monai/data_analysis/datainfo.csv')
    patient_data = datainfo[datainfo['PatientID'] == patientid]
    tracer = patient_data['Tracer'].item()
    diagnosis = patient_data['Diagnosis'].item()

    ct = get_3darray_from_niftipath(ctpath)
    pt = get_3darray_from_niftipath(ptpath)
    gt = get_3darray_from_niftipath(gtpath)
    pr = get_3darray_from_niftipath(prpath)

    dsc = calculate_patient_level_dice_score(gt, pr)
    fnv = calculate_patient_level_false_negative_volume(gt, pr, spacing)
    fpv = calculate_patient_level_false_positive_volume(gt, pr, spacing)

    print('---')
    print(f'{patientid} | {tracer} | {diagnosis}')
    print(f'DSC: {dsc:.4f}, FNV: {fnv:.4f}, FPV: {fpv:.4f}')
    print('---')

    plot_and_save_coronal_sagittal_mip_visualization(ct, pt, gt, pr, patientid, tracer, diagnosis, dsc, fnv, fpv, save_visual_dir)

    return [patientid, tracer, diagnosis, dsc, fnv, fpv]


#%%
def main(args):
    fold = args.fold
    network = args.network_name
    inputsize = args.input_patch_size
    experiment_code = f"{network}_fold{fold}_randcrop{inputsize}_GeneralizedDiceFocalLoss"

    save_preds_dir = os.path.join(RESULTS_FOLDER, f'predictions')
    save_preds_dir = os.path.join(save_preds_dir, 'fold'+str(fold), network, experiment_code) 

    save_metrics_dir = os.path.join(RESULTS_FOLDER, f'metrics')
    save_metrics_dir = os.path.join(save_metrics_dir, 'fold'+str(fold), network, experiment_code) 
    os.makedirs(save_metrics_dir, exist_ok=True)

    save_visual_dir = os.path.join(RESULTS_FOLDER, f'visual')
    save_visual_dir = os.path.join(save_visual_dir, 'fold'+str(fold), network, experiment_code) 
    os.makedirs(save_visual_dir, exist_ok=True)

    valid_data = get_valid_pred_data_in_dict_format(fold, save_preds_dir)
   
    results = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(process_one_case)(data, save_visual_dir) for data in valid_data)

    colnames = ['PatientID', 'Tracer', 'Diagnosis', 'DSC', 'FNV', 'FPV']
    metrics_df = pd.DataFrame(results, columns=colnames)
    metrics_df.to_csv(os.path.join(save_metrics_dir, 'metrics.csv'), index=False)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='AutoPET PET/CT lesion segmentation using MONAI-PyTorch')
    parser.add_argument('--fold', type=int, default=0, metavar='fold',
                        help='validation fold (default: 0), remaining folds will be used for training')
    parser.add_argument('--network-name', type=str, default='unet', metavar='netname',
                        help='network name for training (default: unet)')
    parser.add_argument('--input-patch-size', type=int, default=128, metavar='inputsize',
                        help='size of cropped input patch for training (default: 192)')
    args = parser.parse_args()
    
    main(args)