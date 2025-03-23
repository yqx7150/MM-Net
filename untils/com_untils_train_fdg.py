# import os
# import cv2
import numpy as np
import torch
import tensorflow as tf
from scipy.integrate import trapz

import torch.nn.functional as F
from scipy.io import savemat, loadmat
import torch.fft


def normalize_array(arr):
    """
    最大最小值归一化 对(128,128,18)
    Args:
        arr:

    Returns:

    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
      
        normalize_arr = np.zeros_like(arr)
    else:
        normalize_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalize_arr


def get_mean_k_data_torch(reconstruct_for):
    """
    对numpy类型的数据进行解耦,将(1,48,128,128)的数据-->4 x (1,12,128,128)然后进行平均
    Args:
        reconstruct_for:网络的预测K值(1,48,128,128)

    Returns:预测的k1,k2,k3,k4   4 x (128x128)

    """
   
    k1, k2, k3, k4 = torch.split(reconstruct_for, 12, dim=1)

    pred_k1 = torch.mean(k1.squeeze().double(), dim=0).float()
    pred_k2 = torch.mean(k2.squeeze().double(), dim=0).float()
    pred_k3 = torch.mean(k3.squeeze().double(), dim=0).float()
    pred_k4 = torch.mean(k4.squeeze().double(), dim=0).float()

    return pred_k1, pred_k2, pred_k3, pred_k4

def get_mean_k_data_torch_27(reconstruct_for):
    """
    对numpy类型的数据进行解耦,将(1,48,128,128)的数据-->4 x (1,12,128,128)然后进行平均
    Args:
        reconstruct_for:网络的预测K值(1,48,128,128)

    Returns:预测的k1,k2,k3,k4   4 x (128x128)

    """

    k1, k2, k3, k4 = torch.split(reconstruct_for, 21, dim=1)

    pred_k1 = torch.mean(k1.squeeze().double(), dim=0).float()
    pred_k2 = torch.mean(k2.squeeze().double(), dim=0).float()
    pred_k3 = torch.mean(k3.squeeze().double(), dim=0).float()
    pred_k4 = torch.mean(k4.squeeze().double(), dim=0).float()

    return pred_k1, pred_k2, pred_k3, pred_k4


def get_mean_k_data_torch_92(reconstruct_for):
    """
    对numpy类型的数据进行解耦,将(1,48,128,128)的数据-->4 x (1,12,128,128)然后进行平均
    Args:
        reconstruct_for:网络的预测K值(1,48,128,128)

    Returns:预测的k1,k2,k3,k4   4 x (128x128)

    """
 
    k1, k2, k3, k4 = torch.split(reconstruct_for, 62, dim=1)

    pred_k1 = torch.mean(k1.squeeze().double(), dim=0).float()
    pred_k2 = torch.mean(k2.squeeze().double(), dim=0).float()
    pred_k3 = torch.mean(k3.squeeze().double(), dim=0).float()
    pred_k4 = torch.mean(k4.squeeze().double(), dim=0).float()

    return pred_k1, pred_k2, pred_k3, pred_k4


def compare_ms_ssim(pred, target):
    """
    对数据求ms_ssim指标
    Args:
        pred:
        target:

    Returns:

    """
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


def compute_mean_loss(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的k值和实际值的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """
    pred_k1, pred_k2, pred_k3, pred_k4 = get_mean_k_data_torch(reconstruct_for)
    k1, k2, k3, k4 = get_mean_k_data_torch(target_forward_data)  # torch.Size([1, 12, 128, 128])
    if mse_loss:
        loss_k1 = F.mse_loss(pred_k1, k1)
        loss_k2 = F.mse_loss(pred_k2, k2)
        loss_k3 = F.mse_loss(pred_k3, k3)
        loss_k4 = F.mse_loss(pred_k4, k4)
    else:
        loss_k1 = F.huber_loss(pred_k1, k1)
        loss_k2 = F.huber_loss(pred_k2, k2)
        loss_k3 = F.huber_loss(pred_k3, k3)
        loss_k4 = F.huber_loss(pred_k4, k4)

    return loss_k1, loss_k2, loss_k3, loss_k4


def compute_mean_loss_92(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的k值和实际值的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """
    pred_k1, pred_k2, pred_k3, pred_k4 = get_mean_k_data_torch_92(reconstruct_for)
    k1, k2, k3, k4 = get_mean_k_data_torch_92(target_forward_data)  # torch.Size([1, 12, 128, 128])
    if mse_loss:
        loss_k1 = F.mse_loss(pred_k1, k1)
        loss_k2 = F.mse_loss(pred_k2, k2)
        loss_k3 = F.mse_loss(pred_k3, k3)
        loss_k4 = F.mse_loss(pred_k4, k4)
    else:
        loss_k1 = F.huber_loss(pred_k1, k1)
        loss_k2 = F.huber_loss(pred_k2, k2)
        loss_k3 = F.huber_loss(pred_k3, k3)
        loss_k4 = F.huber_loss(pred_k4, k4)

    return loss_k1, loss_k2, loss_k3, loss_k4


def compute_mean_loss_27(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的k值和实际值的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """
    pred_k1, pred_k2, pred_k3, pred_k4 = get_mean_k_data_torch_27(reconstruct_for)
    k1, k2, k3, k4 = get_mean_k_data_torch_27(target_forward_data)  # torch.Size([1, 12, 128, 128])
    if mse_loss:
        loss_k1 = F.mse_loss(pred_k1, k1)
        loss_k2 = F.mse_loss(pred_k2, k2)
        loss_k3 = F.mse_loss(pred_k3, k3)
        loss_k4 = F.mse_loss(pred_k4, k4)
    else:
        loss_k1 = F.huber_loss(pred_k1, k1)
        loss_k2 = F.huber_loss(pred_k2, k2)
        loss_k3 = F.huber_loss(pred_k3, k3)
        loss_k4 = F.huber_loss(pred_k4, k4)

    return loss_k1, loss_k2, loss_k3, loss_k4

def compute_mean_loss_rev(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的输入值和实际值的输入之间的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """

    pred_data1, pred_data2, pred_data3, pred_data4 = torch.split(reconstruct_for, 12, dim=1)
    if mse_loss:
        rev_loss1 = F.mse_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.mse_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.mse_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.mse_loss(target_forward_data.squeeze(), pred_data4.squeeze())
    else:
        rev_loss1 = F.huber_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.huber_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.huber_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.huber_loss(target_forward_data.squeeze(), pred_data4.squeeze())
   

    return rev_loss1, rev_loss2, rev_loss3, rev_loss4


def compute_mean_loss_rev_92(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的输入值和实际值的输入之间的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """

    pred_data1, pred_data2, pred_data3, pred_data4 = torch.split(reconstruct_for, 62, dim=1)
    if mse_loss:
        rev_loss1 = F.mse_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.mse_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.mse_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.mse_loss(target_forward_data.squeeze(), pred_data4.squeeze())
    else:
        rev_loss1 = F.huber_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.huber_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.huber_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.huber_loss(target_forward_data.squeeze(), pred_data4.squeeze())
   
    return rev_loss1, rev_loss2, rev_loss3, rev_loss4

def compute_mean_loss_rev_27(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的输入值和实际值的输入之间的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """

    pred_data1, pred_data2, pred_data3, pred_data4 = torch.split(reconstruct_for, 21, dim=1)
    if mse_loss:
        rev_loss1 = F.mse_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.mse_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.mse_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.mse_loss(target_forward_data.squeeze(), pred_data4.squeeze())
    else:
        rev_loss1 = F.huber_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.huber_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.huber_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.huber_loss(target_forward_data.squeeze(), pred_data4.squeeze())

    return rev_loss1, rev_loss2, rev_loss3, rev_loss4



def normalize_tensor_torch(tensor):
    """
    最大最小值归一化 对 (128, 128, 18) 的 PyTorch 张量
    Args:
        tensor: PyTorch 张量

    Returns:
        normalize_tensor: 归一化后的 PyTorch 张量
    """
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)

    if tensor_max == tensor_min:
   
        normalize_tensor = torch.zeros_like(tensor)
    else:
        normalize_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    return normalize_tensor


def update_tracer_concentration_torch(reconstruct_for_k_data, cp_data, number):
    """
    由预测的k1,k2,k3,k4生成预测的TAC曲线
    Args:
        reconstruct_for_k_data:1,48,128,128
        cp_data:
        number:

    Returns:

    """
   
    k1, k2, k3, k4 = get_mean_k_data_torch(reconstruct_for_k_data)  
    k4 = torch.zeros(128, 128, dtype=torch.float32).cuda()
   
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
 

    discriminant = torch.maximum(discriminant, torch.tensor(0.0).cuda())
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
    a = a.unsqueeze(2).repeat(1, 1, T)
    b = b.unsqueeze(2).repeat(1, 1, T)
  
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

def update_tracer_concentration_torch_27(reconstruct_for_k_data, cp_data, number):
    """
    由预测的k1,k2,k3,k4生成预测的TAC曲线
    Args:
        reconstruct_for_k_data:1,48,128,128
        cp_data:
        number:

    Returns:

    """
   
    k1, k2, k3, k4 = get_mean_k_data_torch_27(reconstruct_for_k_data)  

    k4 = torch.zeros(128, 128, dtype=torch.float32).cuda()
   
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
  

    discriminant = torch.maximum(discriminant, torch.tensor(0.0).cuda())
  
    alpha1 = (k2 + k3 + k4 - torch.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + torch.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0

    # 计算 a
    a = torch.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])

    # 计算 b
    b = torch.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

    T = len(cp_fmz)
    array = torch.arange(1, T + 1, dtype=torch.float32) 
    array = array.reshape((1, 1, T))  
    a = a.unsqueeze(2).repeat(1, 1, T)
    b = b.unsqueeze(2).repeat(1, 1, T)
  

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


def calculate_xm_torch(tms, tme, CPET, lmbda):
   
    t_values = np.arange(0, CPET.shape[2]) 
    time_indices = np.where((t_values >= tms) & (t_values <= tme))[0]  
    time_indices = torch.from_numpy(time_indices)
    t_values = torch.from_numpy(t_values)
  

    CPET_sub = CPET[:, :, time_indices]  
    t_sub = t_values[time_indices].cuda()  
    lmbda = lmbda.cuda()
 
    integrand = CPET_sub * torch.exp(-lmbda * t_sub)
    xm = torch.trapz(integrand, t_sub)

    return xm



def calculate_img_torch(reconstruct_for_k_data, sampling_intervals, cp_data):
    """
    由CP(t)和CT(t)，采样协议，计算18帧的数据
    :param sampling_intervals:
    :param cp_data:
    :param CT:
    :return:
    """
  
    CP_FMZ = cp_data
    CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
    f_FMZ = torch.zeros((128, 128, 18)).cuda()
    lambda_value = np.log(2) / (109.8 * 60)
    lambda_value = torch.tensor(lambda_value, dtype=torch.float32)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FMZ[:, :, k] = calculate_xm_torch(start_index, end_index, CT_FMZ, lambda_value)
        start_index = end_index + 1
    f_FMZ = f_FMZ / torch.max(f_FMZ)
    return f_FMZ


def calculate_img_torch_27(reconstruct_for_k_data, sampling_intervals, cp_data):
    """
    由CP(t)和CT(t)，采样协议，计算18帧的数据
    :param sampling_intervals:
    :param cp_data:
    :param CT:
    :return:
    """
  
    CP_FMZ = cp_data
    CT_FMZ = update_tracer_concentration_torch_27(reconstruct_for_k_data, CP_FMZ, 0)
    f_FMZ = torch.zeros((128, 128, 27)).cuda()
    lambda_value = np.log(2) / (109.8 * 60)
    lambda_value = torch.tensor(lambda_value, dtype=torch.float32)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FMZ[:, :, k] = calculate_xm_torch(start_index, end_index, CT_FMZ, lambda_value)
        start_index = end_index + 1
    f_FMZ = f_FMZ / torch.max(f_FMZ)
    return f_FMZ


def calculate_img_patlak3_no_weight_92(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            reconstruct_for_k_data:网络输出的k1-k4结果
            target_forward_k_data:实际的k1-k4结果
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_torch(target_forward_k_data, CP_FMZ, 0)

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 92)).cuda()
    y = torch.zeros((128, 128, 92)).cuda()
    base = 0
    for i in range(92):
        index = base + sampling_intervals[i] // 2
        index_end = base + sampling_intervals[i]
        integral_Cp = torch.trapz(CP_FMZ[:, :, :index_end + 1], dim=2)
      
        CT_values_pred = pred_CT_FMZ[:, :, index]
        CP_values_target = CP_FMZ[:, :, index]
     
        mask_x = CP_values_target != 0
      
        x_roi = integral_Cp / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        y_roi = CT_values_pred / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
      
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 52:92]
    nihe_y = y[:, :, 52:92]
 
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

def calculate_img_patlak3_no_weight_27(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            reconstruct_for_k_data:网络输出的k1-k4结果
            target_forward_k_data:实际的k1-k4结果
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch_27(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_torch_27(target_forward_k_data, CP_FMZ, 0)

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 27)).cuda()
    y = torch.zeros((128, 128, 27)).cuda()
    base = 0
    for i in range(27):
        index = base + sampling_intervals[i] // 2
        index_end = base + sampling_intervals[i]
        integral_Cp = torch.trapz(CP_FMZ[:, :, :index_end + 1], dim=2)
       
        CT_values_pred = pred_CT_FMZ[:, :, index]
        CP_values_target = CP_FMZ[:, :, index]
     
        mask_x = CP_values_target != 0
       
        x_roi = integral_Cp / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        y_roi = CT_values_pred / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
       
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 14:27]
    nihe_y = y[:, :, 14:27]
  
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


def calculate_img_patlak3_no_weight(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
   
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2
        index_end = base + sampling_intervals[i]
        integral_Cp = torch.trapz(CP_FMZ[:, :, :index_end + 1], dim=2)
      
        CT_values_pred = pred_CT_FMZ[:, :, index]
        CP_values_target = CP_FMZ[:, :, index]
       
        mask_x = CP_values_target != 0
       
        x_roi = integral_Cp / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        y_roi = CT_values_pred / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
      
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
    #

    # assert 0
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

            # Extract slope (K) and intercept (b)
            K_values[i, j] = coefficients[0].item()
            b_values[i, j] = coefficients[1].item()

    return K_values, b_values


def calculate_img_patlak2_no_weight(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data,
                                    ki_data_batch,
                                    vb_data_batch):

    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
 

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2
        index_end = base + sampling_intervals[i]
        integral_Cp = torch.trapz(CP_FMZ[:, :, :index_end + 1], dim=2)
      
        CT_values_pred = pred_CT_FMZ[:, :, index]
        CP_values_target = CP_FMZ[:, :, index]
      
        mask_x = CP_values_target != 0
       
        x_roi = integral_Cp / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        y_roi = CT_values_pred / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
      
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    x = x[:, :, 8:18]
    y = y[:, :, 8:18]
   

    ki_data_batch = ki_data_batch.permute(1, 2, 0).to(dtype=torch.float32).cuda()
    vb_data_batch = vb_data_batch.permute(1, 2, 0).to(dtype=torch.float32).cuda()
   
    diff_x = torch.zeros((128, 128, 9)).cuda()
    diff_y = torch.zeros((128, 128, 9)).cuda()
    for i in range(0, 9):
        diff_y[:, :, i] = y[:, :, i + 1] - y[:, :, i]
        diff_x[:, :, i] = x[:, :, i + 1] - x[:, :, i]

    target = ki_data_batch * diff_x
    pred = diff_y

   
    pred = pred.float()
   
    target = target.float()

    return pred, target


def calculate_img_patlak3_no_weight_real_data(reconstruct_for_k_data, sampling_intervals, cp_data):
   
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2
        index_end = base + sampling_intervals[i]
        integral_Cp = torch.trapz(CP_FMZ[:, :, :index_end + 1], dim=2)
       
        CT_values_pred = pred_CT_FMZ[:, :, index]
        CP_values_target = CP_FMZ[:, :, index]
       
        mask_x = CP_values_target != 0
        
        x_roi = integral_Cp / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        y_roi = CT_values_pred / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
       
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
   
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


def calculate_img_patlak3_no_weight_test(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
  
    
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    target_CT_FMZ = update_tracer_concentration_torch(target_forward_k_data, CP_FMZ, 0)
    

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2
        index_end = base + sampling_intervals[i]
        integral_Cp = torch.trapz(CP_FMZ[:, :, :index_end + 1], dim=2)
        
        CT_values_target = target_CT_FMZ[:, :, index]
        CP_values_target = CP_FMZ[:, :, index]
       
        mask_x = CP_values_target != 0
      
        x_roi = integral_Cp / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        y_roi = CT_values_target / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
       
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
    

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

def calculate_img_patlak3_no_weight_test_27(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            reconstruct_for_k_data:网络输出的k1-k4结果
            target_forward_k_data:实际的k1-k4结果
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_FMZ = torch.from_numpy(cp_data).cuda()

  
    target_CT_FMZ = update_tracer_concentration_torch_27(target_forward_k_data, CP_FMZ, 0)
   

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 27)).cuda()
    y = torch.zeros((128, 128, 27)).cuda()
    base = 0
    for i in range(27):
        index = base + sampling_intervals[i] // 2
        index_end = base + sampling_intervals[i]
        integral_Cp = torch.trapz(CP_FMZ[:, :, :index_end + 1], dim=2)
       
        CT_values_target = target_CT_FMZ[:, :, index]
        CP_values_target = CP_FMZ[:, :, index]
       
        mask_x = CP_values_target != 0
       
        x_roi = integral_Cp / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        y_roi = CT_values_target / torch.where(mask_x, CP_values_target, torch.ones_like(CP_values_target))
        
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 14:27]
    nihe_y = y[:, :, 14:27]
    #

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

            # Extract slope (K) and intercept (b)
            K_values[i, j] = coefficients[0].item()
            b_values[i, j] = coefficients[1].item()

    return K_values, b_values