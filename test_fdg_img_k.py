import argparse
import os
import numpy as np
import scipy.io as io
import torch
from scipy.io import loadmat
from scipy.integrate import cumtrapz, trapz
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf
from dataset.pet_dataset_fdg import PetDataset
from model.model_new import MultiOutputReversibleGenerator
from untils.com_untils_test import save_img


os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def main():

    CP_PATH = './data/fdg_zubal_head_sample3_kmin_noise_0423/CP/CP_FDG.mat'
    root2 = "./data/fdg_zubal_head_sample3_kmin_noise_0423/test"
    sampling_intervals = [30, 30, 30, 30, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    cp_data = loadmat(CP_PATH)['CP_FDG']
    cp_data = cp_data[:, :3600]
    choose = 1  # 10,11,13

    for checkpoint in range(99, 100):
        checkpoint = '{:04d}'.format(checkpoint)
        
        ckpt = "./exps/ours_fdg_head_sample3_0423_patlak3_no_weight_four_loss/checkpoint/" + str(checkpoint) + ".pth"
        out_path = "./exps/ours_fdg_head_sample3_0423_patlak3_no_weight_four_loss/"
        
        ckpt_allname = ckpt.split("/")[-1]
        batch_size = 1

        net = MultiOutputReversibleGenerator(input_channels=24, output_channels=24, num_blocks=8)
        
        device = torch.device("cuda:0")
        net.to(device) 
        net.eval()
        if os.path.isfile(ckpt):
            net.load_state_dict(torch.load(ckpt), strict=False)  
            print("[INFO] Loaded checkpoint: {}".format(ckpt))

        print("[INFO] Start data load and preprocessing")

        test_dataset = PetDataset(root_folder=root2)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                     drop_last=True)
        PSNR_FDG = []
        SSIM_FDG = []
        MS_SSIM_FDG = []
        MSE_FDG = []
        NRMSE_FDG = []

        PSNR_FDG_IMG = []
        SSIM_FDG_IMG = []
        MS_SSIM_FDG_IMG = []
        MSE_FDG_IMG = []
        NRMSE_FDG_IMG = []

        save_path = out_path + 'test/{}'.format(ckpt_allname)

        print("[INFO] Start test...")
        for i_batch, (
                fdg_batch, fdg_noise_batch, k1_data_batch, k2_data_batch, k3_data_batch, k4_data_batch, ki_data_btach,
                vb_data_batch) in enumerate(
            tqdm(test_dataloader)):  
            k_data_batch = torch.cat(
                (k1_data_batch.squeeze()[:, :, 0].unsqueeze(2), k2_data_batch.squeeze()[:, :, 0].unsqueeze(2),
                 k3_data_batch.squeeze()[:, :, 0].unsqueeze(2),
                 k4_data_batch.squeeze()[:, :, 0].unsqueeze(2)), dim=2)  # 128x128x4
            input_fdg = fdg_noise_batch.permute(0, 3, 1, 2).float().cuda()  #
            target_forward_fdg_label = fdg_batch.permute(0, 3, 1, 2).float().cuda()
            fdg_input = input_fdg[:, 0:12, :, :]
            fdg_input = torch.cat([fdg_input, fdg_input], dim=1)

            with torch.no_grad():  
                reconstruct_for = net(fdg_input)  
            reconstruct_for = torch.abs(reconstruct_for)
            reconstruct_for = torch.clamp(reconstruct_for, 0, 1)  
            k1, k2, k3, k4 = get_mean_k_data(reconstruct_for)
            k4 = torch.zeros_like(k4).cuda()

            pred_k_data = torch.cat(
                (k1.unsqueeze(2), k2.unsqueeze(2), k3.unsqueeze(2), k4.unsqueeze(2)), dim=2)
            pred_fdg_k = pred_k_data.cpu().numpy()
            target_fdg_k = k_data_batch.cpu().numpy()

            for i in range(3):
                psnr_fdg = compare_psnr(255 * abs(pred_fdg_k[:, :, i]), 255 * abs(target_fdg_k[:, :, i]),
                                        data_range=255) 
                ssim_fdg = compare_ssim(abs(target_fdg_k[:, :, i]), abs(pred_fdg_k[:, :, i]), data_range=1)
                mse_fdg = compare_mse(abs(target_fdg_k[:, :, i]), abs(pred_fdg_k[:, :, i]))
                ms_ssim_fdg = compare_ms_ssim(abs(target_fdg_k[:, :, i]), abs(pred_fdg_k[:, :, i]))
                if not np.all(target_fdg_k[:, :, i] == 0):
                    nrmse_fdg = compare_nrmse(abs(target_fdg_k[:, :, i]), abs(pred_fdg_k[:, :, i]))
                else:
                    nrmse_fdg = 0

                PSNR_FDG.append(psnr_fdg)
                SSIM_FDG.append(ssim_fdg)
                MS_SSIM_FDG.append(ms_ssim_fdg)
                MSE_FDG.append(mse_fdg)
                NRMSE_FDG.append(nrmse_fdg)

                os.makedirs(save_path + '/k/pred_fdg_k',
                            exist_ok=True)  
                os.makedirs(save_path + '/k/target_fdg_k', exist_ok=True)
                os.makedirs(save_path + '/k/pred_fdg_k_mat', exist_ok=True)
                os.makedirs(save_path + '/k/target_fdg_k_mat', exist_ok=True)

                save_img(target_fdg_k[:, :, i],
                         save_path + '/k/target_fdg_k' + '/target_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.png')
                save_img(pred_fdg_k[:, :, i],
                         save_path + '/k/pred_fdg_k' + '/pred_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.png')
                # mat
                io.savemat(
                    save_path + '/k/target_fdg_k_mat' + '/target_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.mat',
                    {'data': target_fdg_k[:, :, i]})
                io.savemat(
                    save_path + '/k/pred_fdg_k_mat' + '/pred_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.mat',
                    {'data': pred_fdg_k[:, :, i]})

         
            target_fdg_k = torch.from_numpy(target_fdg_k)
            pred_data = calculate_img_np(reconstruct_for, sampling_intervals, cp_data) 


            target_forward_fdg_label = target_forward_fdg_label.squeeze().permute(1, 2, 0)  
            pred_data = pred_data[:, :, 12:18]
            target_forward_fdg_label = target_forward_fdg_label[:, :, 12:18].cpu().numpy()
            mask = target_forward_fdg_label == 0
            pred_data[mask] = 0
            for i in range(6):
                psnr_fdg_img = compare_psnr(255 * abs(target_forward_fdg_label[:, :, i]), 255 * abs(pred_data[:, :, i]),
                                            data_range=255)  
                ssim_fdg_img = compare_ssim(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]),
                                            data_range=1)
                mse_fdg_img = compare_mse(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]))
                ms_ssim_fdg_img = compare_ms_ssim(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]))
                if not np.all(target_forward_fdg_label[:, :, i] == 0):
                    nrmse_fdg_img = compare_nrmse(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]))
                else:
                    nrmse_fdg_img = 0

                PSNR_FDG_IMG.append(psnr_fdg_img)
                SSIM_FDG_IMG.append(ssim_fdg_img)
                MS_SSIM_FDG_IMG.append(ms_ssim_fdg_img)
                MSE_FDG_IMG.append(mse_fdg_img)
                NRMSE_FDG_IMG.append(nrmse_fdg_img)

                os.makedirs(save_path + '/img/pred_fdg_img',
                            exist_ok=True) 
                os.makedirs(save_path + '/img/target_fdg_img', exist_ok=True)
                os.makedirs(save_path + '/img/pred_fdg_mat', exist_ok=True)
                os.makedirs(save_path + '/img/target_fdg_mat', exist_ok=True)

                save_img(target_forward_fdg_label[:, :, i],
                         save_path + '/img/target_fdg_img' + '/target_fdg_' + str(i_batch + 1) + '_' + str(
                             i + 1) + '.png')
                save_img(pred_data[:, :, i],
                         save_path + '/img/pred_fdg_img' + '/pred_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.png')
                # mat
                io.savemat(
                    save_path + '/img/target_fdg_mat' + '/target_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.mat',
                    {'data': target_forward_fdg_label[:, :, i]})
                io.savemat(
                    save_path + '/img/pred_fdg_mat' + '/pred_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.mat',
                    {'data': pred_data[:, :, i]})

        compare_psnr_show_save(PSNR_FDG, SSIM_FDG, MSE_FDG, MS_SSIM_FDG, NRMSE_FDG,
                               "fdg", save_path, 1, ckpt_allname)
        compare_psnr_show_save(PSNR_FDG_IMG, SSIM_FDG_IMG, MSE_FDG_IMG, MS_SSIM_FDG_IMG, NRMSE_FDG_IMG,
                               "fdg", save_path, 2, ckpt_allname)


def compare_psnr_show_save(PSNR, SSIM, MSE, MS_SSIM, NRMSE, show_name, save_path, index, ckpt_allname):
    print(show_name)
    print(PSNR)
    ave_psnr = sum(PSNR) / len(PSNR)
    PSNR_std = np.std(PSNR)

    ave_ssim = sum(SSIM) / len(SSIM)
    SSIM_std = np.std(SSIM)

    ave_ms_ssim = sum(MS_SSIM) / len(MS_SSIM)
    MS_SSIM_std = np.std(MS_SSIM)

    ave_mse = sum(MSE) / len(MSE)
    MSE_std = np.std(MSE)

    ave_nrmse = sum(NRMSE) / len(NRMSE)
    NRMSE_std = np.std(NRMSE)
    if index == 1:
        file_name = 'k_results_test.txt'
        print('k_results:')
    elif index == 2:
        file_name = 'img_results_test.txt'
        print('img_results:')
    elif index == 3:
        file_name = 'ki_results_test.txt'
        print('ki_results:')
    else:
        file_name = 'vb_results_test.txt'
        print('vb_results:')
    print('ave_psnr_' + show_name, ave_psnr)
    print('ave_ssim_' + show_name, ave_ssim)
    print('ave_ms_ssim_' + show_name, ave_ms_ssim)
    print('ave_mse_' + show_name, ave_mse)
    print('ave_nrmse_' + show_name, ave_nrmse)

    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'a+') as f:
        f.write('\n' * 3)
        f.write(ckpt_allname + '_' + show_name + '\n')

        f.write('ave_psnr:' + str(ave_psnr) + ' ' * 3 + 'PSNR_std:' + str(PSNR_std) + '\n')

        f.write('ave_ssim:' + str(ave_ssim) + ' ' * 3 + 'SSIM_std:' + str(SSIM_std) + '\n')

        f.write('ave_ms_ssim:' + str(ave_ms_ssim) + ' ' * 3 + 'SSIM_std:' + str(MS_SSIM_std) + '\n')

        f.write('ave_mse:' + str(ave_mse) + ' ' * 3 + 'MSE_std:' + str(MSE_std) + '\n')

        f.write('ave_nrmse:' + str(ave_nrmse) + ' ' * 3 + 'nrmse_std:' + str(NRMSE_std) + '\n')


def get_mean_k_data(reconstruct_for):
    """
    # get_mean_k_data_np的tensor版本
    Args:
        reconstruct_for:

    Returns:

    """
    k1, k2, k3, k4 = torch.split(reconstruct_for, 12, dim=1)
    pred_k1 = torch.mean(k1.squeeze(), dim=0)
    pred_k2 = torch.mean(k2.squeeze(), dim=0)
    pred_k3 = torch.mean(k3.squeeze(), dim=0)
    pred_k4 = torch.mean(k4.squeeze(), dim=0)
    return pred_k1, pred_k2, pred_k3, pred_k4


def calculate_img_np(reconstruct_for_k_data, sampling_intervals, cp_data):
    
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().detach().numpy() 

    CP_FDG = cp_data
    pred_CT_FDG = update_tracer_concentration_np(reconstruct_for_k_data, CP_FDG)

    f_FDG = np.zeros((128, 128, 18))

    lambda_value = np.log(2) / (109.8 * 60)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FDG[:, :, k] = calculate_xm(start_index, end_index, pred_CT_FDG, lambda_value)
        start_index = end_index + 1

    pred_fdg = f_FDG / np.max(f_FDG)
    return pred_fdg


def get_mean_k_data_np(reconstruct_for):
    k1, k2, k3, k4 = np.split(reconstruct_for, 4, axis=1)
    pred_k1 = np.mean(np.squeeze(k1).astype(np.float64), axis=0).astype(np.float32)
    pred_k2 = np.mean(np.squeeze(k2).astype(np.float64), axis=0).astype(np.float32)
    pred_k3 = np.mean(np.squeeze(k3).astype(np.float64), axis=0).astype(np.float32)
    pred_k4 = np.mean(np.squeeze(k4).astype(np.float64), axis=0).astype(np.float32)
    return pred_k1, pred_k2, pred_k3, pred_k4


def compare_ms_ssim(pred, target):
   
    pred = tf.convert_to_tensor(pred)
    pred = tf.expand_dims(pred, axis=-1)

    target = tf.convert_to_tensor(target)
    target = tf.expand_dims(target, axis=-1)
    pred = tf.cast(pred, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)
    max_val = 1

    result = tf.image.ssim_multiscale(pred, target, max_val, filter_size=1)
    result = result.numpy()
    return result


def update_tracer_concentration_np(reconstruct_for_k_data, cp_data):
   

    k1, k2, k3, k4 = get_mean_k_data_np(reconstruct_for_k_data)  

    k1 = k1.squeeze()
    k2 = k2.squeeze()
    k3 = k3.squeeze()

    k4 = np.zeros((128, 128), dtype=np.float32)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    cp_fdg = np.array(cp_data[0].tolist())
    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4
    discriminant = np.maximum(discriminant, 0)  
    alpha1 = (k2 + k3 + k4 - np.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + np.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0
    a = np.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])
    b = np.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

   
    T = len(cp_fdg)
    array = np.arange(1, T + 1) 
    array = array.reshape((1, 1, T)) 
    a = np.repeat(a[:, :, np.newaxis], T, axis=2)  
    b = np.repeat(b[:, :, np.newaxis], T, axis=2)  

    alpha1 = np.repeat(alpha1[:, :, np.newaxis], T, axis=2)
    alpha2 = np.repeat(alpha2[:, :, np.newaxis], T, axis=2)
    part11 = a * cp_fdg  
    part12 = np.exp(-alpha1 * array)  

    part21 = b * cp_fdg  
    part22 = np.exp(-alpha2 * array)  

    
    temp_part11 = np.fft.fft(part11)
    temp_part12 = np.fft.fft(part12)
    CT1_temp = np.fft.ifft(temp_part11 * temp_part12)
    CT1_temp = np.real(CT1_temp)
    CT1 = CT1_temp[:, :, :T]

    temp_part21 = np.fft.fft(part21)
    temp_part22 = np.fft.fft(part22)
    CT2_temp = np.fft.ifft(temp_part21 * temp_part22)
    CT2_temp = np.real(CT2_temp)
    CT2 = CT2_temp[:, :, :T]
    CT = CT1 + CT2

    return CT


def calculate_xm(tms, tme, CPET, lmbda):
  
    t_values = np.arange(0, CPET.shape[2])  

    time_indices = np.where((t_values >= tms) & (t_values <= tme))[0] 
    CPET_sub = CPET[:, :, time_indices] 
    t_sub = t_values[time_indices]  

    integrand = CPET_sub * np.exp(-lmbda * t_sub)
    xm = trapz(integrand, t_sub)

    return xm


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
