import os
import cv2
import numpy as np
import openpyxl
import torch
import tensorflow as tf
from scipy import io
from scipy.integrate import cumtrapz, trapz
from scipy.signal import fftconvolve
import torch.nn.functional as F
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from openpyxl import Workbook


def get_mean_k_data_np(reconstruct_for):
    k1, k2, k3, k4 = np.split(reconstruct_for, 4, axis=1)
    pred_k1 = np.mean(k1.squeeze(), axis=0)
    pred_k2 = np.mean(k2.squeeze(), axis=0)
    pred_k3 = np.mean(k3.squeeze(), axis=0)
    pred_k4 = np.mean(k4.squeeze(), axis=0)
    return pred_k1, pred_k2, pred_k3, pred_k4


def process_and_save_metrics(pred_data, target_data, sheet, show_name, save_path, excel_save_path, i_batch, workbook):
    PSNR_FMZ = []
    SSIM_FMZ = []
    MS_SSIM_FMZ = []
    MSE_FMZ = []
    NRMSE_FMZ = []
    # print(i_batch)
    # assert 0
    num = target_data.shape[-1]
    for i in range(num):
        psnr_fmz = compare_psnr(255 * np.abs(pred_data[:, :, i]), 255 * np.abs(target_data[:, :, i]), data_range=255)
        ssim_fmz = compare_ssim(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]), data_range=1)
        mse_fmz = compare_mse(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]))
        ms_ssim_fmz = compare_ms_ssim(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]))

        if not np.all(target_data[:, :, i] == 0):
            nrmse_fmz = compare_nrmse(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]))
        else:
            nrmse_fmz = 0

        PSNR_FMZ.append(psnr_fmz)
        SSIM_FMZ.append(ssim_fmz)
        MS_SSIM_FMZ.append(ms_ssim_fmz)
        MSE_FMZ.append(mse_fmz)
        NRMSE_FMZ.append(nrmse_fmz)

        os.makedirs(save_path + '/pred_data_' + show_name + '_img', exist_ok=True)
        os.makedirs(save_path + '/target_data_' + show_name + '_img', exist_ok=True)
        os.makedirs(save_path + '/pred_data_' + show_name + '_mat', exist_ok=True)
        os.makedirs(save_path + '/target_data_' + show_name + '_mat', exist_ok=True)

        save_img(target_data[:, :, i],
                 save_path+'/target_data_' + show_name + '_img/target_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.png')
        save_img(pred_data[:, :, i], save_path+'/pred_data_' + show_name + '_img/pred_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.png')

        io.savemat(save_path+'/target_data_' + show_name + '_mat/target_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.mat',
                   {'data': target_data[:, :, i]})
        io.savemat(save_path+'/pred_data_' + show_name + '_mat/pred_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.mat',
                   {'data': pred_data[:, :, i]})

        data = ['fmz_'+str(i_batch + 1)+'_'+str(i + 1), round(psnr_fmz, 4), round(ssim_fmz, 4),
                round(ms_ssim_fmz, 4), round(mse_fmz, 4), round(nrmse_fmz, 4)]

        sheet.append(data)
        workbook.save(excel_save_path)
    return PSNR_FMZ, SSIM_FMZ, MS_SSIM_FMZ, MSE_FMZ, NRMSE_FMZ



def generate_k_and_b(reconstruct_for_k_data, sampling_intervals, CP_FMZ):
   
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().numpy()
    CT_values = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)
    Cp_values = CP_FMZ
  
    sample_all_time = np.sum(sampling_intervals)
    Cp_values = np.tile(Cp_values, (128, 128, 1))
    sample_time = np.arange(0, sample_all_time, 1)
    Cp_values = Cp_values[:, :, :3600]
    sample_time = sample_time[:3600]
    CT_values = CT_values[:, :, :3600]
    integral_Cp = cumtrapz(Cp_values, sample_time, axis=2, initial=0)
    integral_CT = cumtrapz(CT_values, sample_time, axis=2, initial=0)
    x_roi = np.divide(integral_Cp, CT_values, out=np.zeros_like(integral_Cp), where=CT_values != 0)
    y_roi = np.divide(integral_CT, CT_values, out=np.zeros_like(integral_CT), where=CT_values != 0)
    x = x_roi[:, :, 600:3600] 
    y = y_roi[:, :, 600:3600]  
    rows, columns, time_points = x.shape

   
    K_values = np.zeros((rows, columns))
    b_values = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
           
            x_flat = x[i, j, :].reshape(-1)
            y_flat = y[i, j, :].reshape(-1)

           
            if np.all(x_flat == 0) or np.all(y_flat == 0):
                continue

           

            linear_fit = np.polyfit(x_flat, y_flat, 1)

           
            K_values[i, j] = linear_fit[0]
            b_values[i, j] = linear_fit[1]

    return K_values, b_values

def generate_k_and_b_logan(reconstruct_for_k_data, sampling_intervals, CP_FMZ):
  
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().numpy()
    CT_values = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)
    Cp_values = np.array(CP_FMZ[0].tolist())
   
    sample_all_time = np.sum(sampling_intervals)
    Cp_values = np.tile(Cp_values, (128, 128, 1))

    x = np.zeros((128, 128, 18))
    y = np.zeros((128, 128, 18))
    base = 0
    for i in range(0, 18):
        
        index = base + sampling_intervals[i] // 2
        index_intergral = base + sampling_intervals[i]
       
        integral_Cp = np.trapz(Cp_values[:, :, :index + 1], axis=2)
        integral_CT = np.trapz(CT_values[:, :, :index + 1], axis=2)
      
        CT_values1 = CT_values[:, :, index]
        x_roi = np.divide(integral_Cp, CT_values1, out=np.zeros_like(integral_Cp), where=CT_values1 != 0)
        y_roi = np.divide(integral_CT, CT_values1, out=np.zeros_like(integral_CT), where=CT_values1 != 0)
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

   
    rows, columns, time_points = x.shape

   
    K_values = np.zeros((rows, columns))
    b_values = np.zeros((rows, columns))
    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
    for i in range(rows):
        for j in range(columns):
         
            x_flat = nihe_x[i, j, :].reshape(-1)
            y_flat = nihe_y[i, j, :].reshape(-1)

          
            if np.all(x_flat == 0) or np.all(y_flat == 0):
                continue

          
            linear_fit = np.polyfit(x_flat, y_flat, 1)

           
            K_values[i, j] = linear_fit[0]
            b_values[i, j] = linear_fit[1]

    return K_values, b_values

def compare_psnr_show_save(PSNR, SSIM, MSE, MS_SSIM, NRMSE, show_name, save_path, index, ckpt_allname):
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


def save_img_color(img, img_path):
    """

    Args:
        img:
        img_path:

    Returns:

    """
    img = np.clip(img * 255, 0, 255)

    img_1 = img[:, :,
            :: -1] 
    cv2.imwrite(img_path, img_1)


def save_img(img, img_path):
    """
    保存img为白底
    Args:
        img:
        img_path:

    Returns:

    """
    img = np.clip(img * 255, 0,
                  255)
    img = 255 - img  

    cv2.imwrite(img_path, img)



def normalize_array(arr):
    
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
       
        normalize_arr = np.zeros_like(arr)
    else:
        normalize_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalize_arr


def get_mean_k_data_torch(reconstruct_for):
   
    k1, k2, k3, k4 = torch.split(reconstruct_for, 12, dim=1)
    pred_k1 = torch.mean(k1.squeeze(), dim=0)
    pred_k2 = torch.mean(k2.squeeze(), dim=0)
    pred_k3 = torch.mean(k3.squeeze(), dim=0)
    pred_k4 = torch.mean(k4.squeeze(), dim=0)

    return pred_k1, pred_k2, pred_k3, pred_k4


def get_mean_k_data(reconstruct_for):
   
    k1, k2, k3, k4 = torch.split(reconstruct_for, 12, dim=1)
    pred_k1 = torch.mean(k1.squeeze(), dim=0)
    pred_k2 = torch.mean(k2.squeeze(), dim=0)
    pred_k3 = torch.mean(k3.squeeze(), dim=0)
    pred_k4 = torch.mean(k4.squeeze(), dim=0)
    return pred_k1, pred_k2, pred_k3, pred_k4


def get_mean_k_data_27(reconstruct_for):
  
    k1, k2, k3, k4 = torch.split(reconstruct_for, 21, dim=1)
    pred_k1 = torch.mean(k1.squeeze(), dim=0)
    pred_k2 = torch.mean(k2.squeeze(), dim=0)
    pred_k3 = torch.mean(k3.squeeze(), dim=0)
    pred_k4 = torch.mean(k4.squeeze(), dim=0)
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


def compute_mean_loss(reconstruct_for, target_forward_data):
   
    pred_k1, pred_k2, pred_k3, pred_k4 = get_mean_k_data(reconstruct_for)
    k1, k2, k3, k4 = get_mean_k_data(target_forward_data)  
    loss_k1 = F.mse_loss(pred_k1, k1)
    loss_k2 = F.mse_loss(pred_k2, k2)
    loss_k3 = F.mse_loss(pred_k3, k3)
    loss_k4 = F.mse_loss(pred_k4, k4)
    return loss_k1, loss_k2, loss_k3, loss_k4


def compute_mean_loss_rev(reconstruct_for, target_forward_data):
  

    pred_data1, pred_data2, pred_data3, pred_data4 = torch.split(reconstruct_for, 12, dim=1)

    rev_loss1 = F.mse_loss(target_forward_data.squeeze(), pred_data1.squeeze())
    rev_loss2 = F.mse_loss(target_forward_data.squeeze(), pred_data2.squeeze())
    rev_loss3 = F.mse_loss(target_forward_data.squeeze(), pred_data3.squeeze())
    rev_loss4 = F.mse_loss(target_forward_data.squeeze(), pred_data4.squeeze())
    return rev_loss1, rev_loss2, rev_loss3, rev_loss4


def normalize_tensor_torch(tensor):
  
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)

    if tensor_max == tensor_min:
       
        normalize_tensor = torch.zeros_like(tensor)
    else:
        normalize_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    return normalize_tensor


def update_tracer_concentration_np(reconstruct_for_k_data, cp_data, number):
   
    k1, k2, k3, k4 = get_mean_k_data_np(reconstruct_for_k_data)  # 128x128
    

    
    if number == 1:
        k1 = normalize_array(k1)
        k2 = normalize_array(k2)
        k3 = normalize_array(k3)
        k4 = normalize_array(k4)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    cp_fmz = np.array(cp_data[0].tolist())
    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4
    discriminant = np.maximum(discriminant, 0)  # 将负值替换为零
    alpha1 = (k2 + k3 + k4 - np.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + np.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0
   
    a = np.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])
    
    b = np.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

   
    T = len(cp_fmz)
    array = np.arange(1, T + 1) 
    array = array.reshape((1, 1, T)) 
    a = np.repeat(a[:, :, np.newaxis], T, axis=2)  
    b = np.repeat(b[:, :, np.newaxis], T, axis=2) 

    alpha1 = np.repeat(alpha1[:, :, np.newaxis], T, axis=2)
    alpha2 = np.repeat(alpha2[:, :, np.newaxis], T, axis=2)
    part11 = a * cp_fmz 
    part12 = np.exp(-alpha1 * array)  

    part21 = b * cp_fmz 
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


def calculate_img_np(reconstruct_for_k_data, sampling_intervals, cp_data):
  
    
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().detach().numpy()  

    CP_FMZ = cp_data
    pred_CT_FMZ = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)

    f_FMZ = np.zeros((128, 128, 18))

    lambda_value = np.log(2) / (20.4 * 60)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FMZ[:, :, k] = calculate_xm(start_index, end_index, pred_CT_FMZ, lambda_value)
        start_index = end_index + 1

    pred_fmz = f_FMZ / np.max(f_FMZ)
   
    return pred_fmz


def calculate_img_np27(reconstruct_for_k_data, sampling_intervals, cp_data):
    
  
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().detach().numpy()  

    CP_FMZ = cp_data
    pred_CT_FMZ = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)

    f_FMZ = np.zeros((128, 128, 27))

    lambda_value = np.log(2) / (109.8 * 60)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FMZ[:, :, k] = calculate_xm(start_index, end_index, pred_CT_FMZ, lambda_value)
        start_index = end_index + 1

    pred_fmz = f_FMZ / np.max(f_FMZ)
   
    return pred_fmz


def calculate_logan_img(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data, ki_data_btach,
                        vb_data_batch):
  
    CP_FMZ = cp_data
    target_forward_k_data = target_forward_k_data.detach().cpu().numpy()
    reconstruct_for_k_data = reconstruct_for_k_data.detach().cpu().numpy()
    ki_data_btach = ki_data_btach.detach().cpu().numpy()
    vb_data_batch = vb_data_batch.detach().cpu().numpy()
   

    pred_CT_FMZ = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_np(target_forward_k_data, CP_FMZ, 0)

   
    integral_Cp = np.cumsum(CP_FMZ, axis=1) 
    integral_Cp = np.tile(integral_Cp, (128, 128, 1))
    pred_integral_CT = np.cumsum(pred_CT_FMZ, axis=2)
 
  
    x_roi = np.divide(integral_Cp, target_CT_FMZ, where=target_CT_FMZ != 0, out=np.zeros_like(target_CT_FMZ))

  
    y_roi = np.divide(pred_integral_CT, pred_CT_FMZ, where=pred_CT_FMZ != 0, out=np.zeros_like(pred_CT_FMZ))
   

    x = x_roi[:, :, 600:3600] 
    y = y_roi[:, :, 600:3600]
  
    ki_data_btach = np.moveaxis(ki_data_btach, 0, -1)  
    vb_data_batch = np.moveaxis(vb_data_batch, 0, -1)
  
    target = x * ki_data_btach + vb_data_batch
    pred = y

    pred = pred / np.max(pred)
    target = target / np.max(target)

  
    pred = torch.from_numpy(pred).cuda()
    target = torch.from_numpy(target).cuda()
    return pred, target

def update_tracer_concentration_torch(reconstruct_for_k_data, cp_data, number):
   
  
    k1 = reconstruct_for_k_data[:, :, 0]
    k2 = reconstruct_for_k_data[:, :, 1]
    k3 = reconstruct_for_k_data[:, :, 2]
    k4 = reconstruct_for_k_data[:, :, 3]

   
    if number == 1:
        k1 = normalize_tensor_torch(k1)
        k2 = normalize_tensor_torch(k2)
        k3 = normalize_tensor_torch(k3)
        k4 = normalize_tensor_torch(k4)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    cp_fmz = torch.tensor(cp_data[0].tolist(), dtype=torch.float32)
    cp_fmz = cp_fmz.float().cuda()


    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4
  
    discriminant = torch.maximum(discriminant, torch.tensor(0.0))
    
    alpha1 = (k2 + k3 + k4 - torch.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + torch.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0

    
    a = torch.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])


    b = torch.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

    T = len(cp_fmz)
    array = torch.arange(1, T + 1, dtype=torch.float32)  
    array = array.reshape((1, 1, T))  
    a = a.unsqueeze(2).repeat(1, 1, T).cuda()
    b = b.unsqueeze(2).repeat(1, 1, T).cuda()
   
    alpha1 = alpha1.unsqueeze(2).repeat(1, 1, T)
    alpha2 = alpha2.unsqueeze(2).repeat(1, 1, T)

    array = array.cuda()
    part11 = a * cp_fmz  
    part12 = torch.exp(-alpha1 * array) 

    part21 = b * cp_fmz  
    part22 = torch.exp(-alpha2 * array)  

  
    temp_part11 = torch.fft.fft(part11)
    temp_part12 = torch.fft.fft(part12)
    CT1_temp = torch.fft.ifft(temp_part11 * temp_part12)
    CT1_temp = torch.real(CT1_temp)
    CT1 = CT1_temp[:, :, :T]

    temp_part21 = torch.fft.fft(part21)
    temp_part22 = torch.fft.fft(part22)
    CT2_temp = torch.fft.ifft(temp_part21 * temp_part22)
    CT2_temp = torch.real(CT2_temp)
    CT2 = CT2_temp[:, :, :T]
    CT = CT1 + CT2

    return CT


def calculate_img_logan3_no_weight_test(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
   
    CP_FMZ = torch.from_numpy(cp_data).cuda()

   
    target_CT_FMZ = update_tracer_concentration_torch(target_forward_k_data, CP_FMZ, 0)

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2

        integral_Cp = torch.trapz(CP_FMZ[:, :, :index + 1], dim=2)
       
        integral_target_CT = torch.trapz(target_CT_FMZ[:, :, :index + 1], dim=2)
        CT_values_target = target_CT_FMZ[:, :, index]
       
        mask_x = CT_values_target != 0
       
        x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        y_roi = integral_target_CT / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))

        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 12:18]
    nihe_y = y[:, :, 12:18]


    rows, columns, time_points = x.shape
    K_values = torch.zeros((rows, columns)).cuda()
    b_values = torch.zeros((rows, columns)).cuda()

    for i in range(rows):
        for j in range(columns):
          
            x_flat = nihe_x[i, j, :].view(-1)
            y_flat = nihe_y[i, j, :].view(-1)

           
            if torch.all(x_flat == 0) or torch.all(y_flat == 0):
                continue

          
            X_augmented = torch.stack([x_flat, torch.ones_like(x_flat)], dim=1)
           
            coefficients = torch.linalg.lstsq(X_augmented, y_flat.view(-1, 1)).solution

        
            K_values[i, j] = coefficients[0].item()
            b_values[i, j] = coefficients[1].item()

    return K_values, b_values