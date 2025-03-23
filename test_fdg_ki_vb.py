import argparse
import os
import cv2
import numpy as np
import scipy.io as io
import torch
from PIL import Image
import tensorflow as tf
from scipy.io import loadmat
from scipy.integrate import cumtrapz, trapz
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.pet_dataset_fdg import PetDataset
from model.model_new import MultiOutputReversibleGenerator
from untils.com_untils_train_fdg import calculate_img_patlak3_no_weight, calculate_img_patlak3_no_weight_test

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    # ======================================define the model============================================
    # 配置信息
    CP_PATH = './data/fdg_zubal_head_sample3_kmin_noise_0423/CP/CP_FDG.mat'
    root2 = "./data/fdg_zubal_head_sample3_kmin_noise_0423/test"
    sampling_intervals = [30, 30, 30, 30, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    cp_data = loadmat(CP_PATH)['CP_FDG']
    cp_data = cp_data[:, :3600]

    for checkpoint in range(99, 100):
        checkpoint = '{:04d}'.format(checkpoint)
        
        ckpt = "./exps/ours_fdg_head_sample3_0423_patlak3_no_weight/checkpoint/" + str(checkpoint) + ".pth"
        out_path = "./exps/ours_fdg_head_sample3_0423_patlak3_no_weight/"
        

        ckpt_allname = ckpt.split("/")[-1]
        batch_size = 1

        net = MultiOutputReversibleGenerator(input_channels=24, output_channels=24, num_blocks=8)
        device = torch.device("cuda:0")
        net.to(device)  # 将模型加载到相应的设备中
        net.eval()
        # load the pretrained weight if there exists one
        
        if os.path.isfile(ckpt):
            net.load_state_dict(torch.load(ckpt), strict=False)  # 加载模型参数
            print("[INFO] Loaded checkpoint: {}".format(ckpt))

        print("[INFO] Start data load and preprocessing")

        test_dataset = PetDataset(root_folder=root2)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                     drop_last=True)
        PSNR_FDG_KI = []
        SSIM_FDG_KI = []
        MS_SSIM_FDG_KI = []
        MSE_FDG_KI = []
        NRMSE_FDG_KI = []

        PSNR_FDG_VB = []
        SSIM_FDG_VB = []
        MS_SSIM_FDG_VB = []
        MSE_FDG_VB = []
        NRMSE_FDG_VB = []

        save_path = out_path + 'test/{}'.format(ckpt_allname)

        print("[INFO] Start test...")
        for i_batch, (fdg_batch, fdg_noise_batch, k1_data_batch, k2_data_batch, k3_data_batch, k4_data_batch, ki_data_btach,
                      vb_data_batch) in enumerate(
            tqdm(test_dataloader)):  # tqdm是一个可以显示进度条的模块。enumerate()函数是python的内置函数，可以同时遍历lt中元素及其索引，i是索引，item是lt中的元素。
            # k_data_batch = torch.cat(
            #     (k1_data_batch.squeeze()[:, :, 0].unsqueeze(2), k2_data_batch.squeeze()[:, :, 0].unsqueeze(2),
            #      k3_data_batch.squeeze()[:, :, 0].unsqueeze(2),
            #      k4_data_batch.squeeze()[:, :, 0].unsqueeze(2)), dim=2)  # 128x128x4
    
            target_forward_k1_data = k1_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]
            target_forward_k2_data = k2_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]
            target_forward_k3_data = k3_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]
            target_forward_k4_data = k4_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]
            target_forward_k_data = torch.cat(  # [1,48,128,128]
                (target_forward_k1_data, target_forward_k2_data, target_forward_k3_data, target_forward_k4_data), dim=1)
            target_forward_fdg = fdg_noise_batch.permute(0, 3, 1, 2).float().cuda()  #
            target_forward_fdg_label = fdg_batch.permute(0, 3, 1, 2).float().cuda()
            fdg_input = target_forward_fdg[:, 0:12, :, :]
            fdg_input = torch.cat([fdg_input, fdg_input], dim=1)
    
            with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
                reconstruct_for = net(fdg_input)  # hybrid_input [1,2,18,128,128]
            reconstruct_for = torch.abs(reconstruct_for)
            reconstruct_for = torch.clamp(reconstruct_for, 0, 1)  # torch.Size([1, 96, 128, 128])
            # k1, k2, k3, k4 = get_mean_k_data(reconstruct_for)
            # pred_k_data = torch.cat(
            #     (k1.unsqueeze(2), k2.unsqueeze(2), k3.unsqueeze(2), k4.unsqueeze(2)), dim=2)
            # pred_fdg_k = pred_k_data.cpu().numpy()
            # target_fdg_k = k_data_batch.cpu().numpy()
    
            ## ki img
    
            pred_ki, pred_vb = calculate_img_patlak3_no_weight(reconstruct_for, target_forward_k_data, sampling_intervals,
                                                              cp_data)
            # print(target_forward_k_data.shape)
            # assert 0
            # pred_ki, pred_vb = calculate_img_patlak3_no_weight_test(reconstruct_for, target_forward_k_data, sampling_intervals,
            #                                                     cp_data)
    
            target_forward_ki_label = ki_data_btach.squeeze().numpy()  # 128x128
            target_forward_vb_label = vb_data_batch.squeeze().numpy()  # 128x128
            target_forward_vb_label = abs(target_forward_vb_label)
            pred_ki_data = pred_ki.cpu().numpy()
            pred_vb_data = pred_vb.cpu().numpy()
            pred_vb_data = abs(pred_vb_data)
            mask_ki = target_forward_ki_label == 0
            mask_vb = target_forward_vb_label == 0
    
            pred_ki_data[mask_ki] = 0
            pred_vb_data[mask_vb] = 0
            
            pred_ki_data = normalize_array(pred_ki_data)
            pred_vb_data = normalize_array(pred_vb_data)
            target_forward_ki_label = normalize_array(target_forward_ki_label)
            target_forward_vb_label = normalize_array(target_forward_vb_label)
    
            ## ki img
            psnr_fdg_ki = compare_psnr(target_forward_ki_label, pred_ki_data,
                                       data_range=np.max(target_forward_ki_label))  # )函数返回数字的绝对值。
            # psnr_fdg_ki = 0
            # ssim_fdg_ki = 0
            ssim_fdg_ki = compare_ssim(target_forward_ki_label, pred_ki_data, data_range=np.max(target_forward_ki_label))
            mse_fdg_ki = compare_mse(target_forward_ki_label, pred_ki_data)
            ms_ssim_fdg_ki = compare_ms_ssim(abs(target_forward_ki_label), abs(pred_ki_data))
            if not np.all(target_forward_ki_label == 0):
                nrmse_fdg_ki = compare_nrmse(target_forward_ki_label, pred_ki_data)
            else:
                nrmse_fdg_ki = 0
    
            PSNR_FDG_KI.append(psnr_fdg_ki)
            SSIM_FDG_KI.append(ssim_fdg_ki)
            MS_SSIM_FDG_KI.append(ms_ssim_fdg_ki)
            MSE_FDG_KI.append(mse_fdg_ki)
            NRMSE_FDG_KI.append(nrmse_fdg_ki)
    
            #  保存 result
            os.makedirs(save_path + '/ki/pred_fdg_ki',
                        exist_ok=True)  # 递归创建目录，题中应有之意即路径中哪一层不存在，则自动创建。如果exist_ok是False（默认），当目标目录（即要创建的目录）已经存在，会抛出一个OSError。
            os.makedirs(save_path + '/ki/target_fdg_ki', exist_ok=True)
            os.makedirs(save_path + '/ki/pred_fdg_ki_mat', exist_ok=True)
            os.makedirs(save_path + '/ki/target_fdg_ki_mat', exist_ok=True)
    
            save_img(target_forward_ki_label,
                     save_path + '/ki/target_fdg_ki' + '/target_fdg_' + str(i_batch + 1) + '.png')
            save_img(pred_ki_data,
                     save_path + '/ki/pred_fdg_ki' + '/pred_fdg_' + str(i_batch + 1) + '.png')
            # mat
            io.savemat(save_path + '/ki/target_fdg_ki_mat' + '/target_fdg_' + str(i_batch + 1) + '.mat',
                       {'data': target_forward_ki_label})
            io.savemat(save_path + '/ki/pred_fdg_ki_mat' + '/pred_fdg_' + str(i_batch + 1) + '.mat',
                       {'data': pred_ki_data})
    
            ## vb img
            psnr_fdg_vb = compare_psnr(abs(target_forward_vb_label), abs(pred_vb_data),
                                       data_range=abs(np.max(target_forward_vb_label)))  # abs()函数返回数字的绝对值。
            ssim_fdg_vb = compare_ssim(abs(target_forward_vb_label), abs(pred_vb_data),
                                       data_range=abs(np.max(target_forward_vb_label)))
            mse_fdg_vb = compare_mse(target_forward_vb_label, pred_vb_data)
            ms_ssim_fdg_vb = compare_ms_ssim(abs(target_forward_vb_label), abs(pred_vb_data))
            if not np.all(target_forward_vb_label == 0):
                nrmse_fdg_vb = compare_nrmse(target_forward_vb_label, pred_vb_data)
            else:
                nrmse_fdg_vb = 0
    
            PSNR_FDG_VB.append(psnr_fdg_vb)
            SSIM_FDG_VB.append(ssim_fdg_vb)
            MS_SSIM_FDG_VB.append(ms_ssim_fdg_vb)
            MSE_FDG_VB.append(mse_fdg_vb)
            NRMSE_FDG_VB.append(nrmse_fdg_vb)
    
            #  保存 result
            os.makedirs(save_path + '/vb/pred_fdg_vb',
                        exist_ok=True)  # 递归创建目录，题中应有之意即路径中哪一层不存在，则自动创建。如果exist_ok是False（默认），当目标目录（即要创建的目录）已经存在，会抛出一个OSError。
            os.makedirs(save_path + '/vb/target_fdg_vb', exist_ok=True)
            os.makedirs(save_path + '/vb/pred_fdg_vb_mat', exist_ok=True)
            os.makedirs(save_path + '/vb/target_fdg_vb_mat', exist_ok=True)
    
            save_img_vb(target_forward_vb_label,
                        save_path + '/vb/target_fdg_vb' + '/target_fdg_' + str(i_batch + 1) + '.png')
            save_img_vb(pred_vb_data,
                        save_path + '/vb/pred_fdg_vb' + '/pred_fdg_' + str(i_batch + 1) + '.png')
            # mat
            io.savemat(save_path + '/vb/target_fdg_vb_mat' + '/target_fdg_' + str(i_batch + 1) + '.mat',
                       {'data': target_forward_vb_label})
            io.savemat(save_path + '/vb/pred_fdg_vb_mat' + '/pred_fdg_' + str(i_batch + 1) + '.mat',
                       {'data': pred_vb_data})
    
        compare_psnr_show_save(PSNR_FDG_KI, SSIM_FDG_KI, MSE_FDG_KI, MS_SSIM_FDG_KI, NRMSE_FDG_KI,
                               "fdg", save_path, 3, ckpt_allname)
        compare_psnr_show_save(PSNR_FDG_VB, SSIM_FDG_VB, MSE_FDG_VB, MS_SSIM_FDG_VB, NRMSE_FDG_VB,
                               "fdg", save_path, 4, ckpt_allname)


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
def save_img(img, img_path):
    """
    保存img为白底
    Args:
        img:
        img_path:

    Returns:

    """
    img = np.clip(img * 255, 0,
                  255)  # np.clip(a,a_min,a_max,out=None)是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。该函数的作用是将数组a中的所有数限定到范围a_min和a_max中。a：输入矩阵；a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；out：可以指定输出矩阵的对象，shape与a相同
    img = 255 - img  # 保存为白色为底色的图片

    cv2.imwrite(img_path, img)
    # 将矩阵转换为图像
    # img = Image.fromarray((img).astype(np.uint8))
    #
    # # 创建白色背景图像，并将矩阵图像粘贴在上面
    # # white_img = Image.new('RGB', (128, 128), color='white')
    #
    # # white_img.paste(img, (0, 0), img)
    #
    # # 保存图像
    # img.save(img_path)


def save_img_vb(img, img_path):
    """
    保存img为白底
    Args:
        img:
        img_path:

    Returns:

    """
    img = np.clip(img * 255, 0,
                  255)  # np.clip(a,a_min,a_max,out=None)是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。该函数的作用是将数组a中的所有数限定到范围a_min和a_max中。a：输入矩阵；a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；out：可以指定输出矩阵的对象，shape与a相同
    img = 255 - img  # 保存为白色为底色的图片

    cv2.imwrite(img_path, img)


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
        # 分母为零，避免除以零的情况
        normalize_arr = np.zeros_like(arr)
    else:
        normalize_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalize_arr


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
    """

    Args:
        reconstruct_for_k_data: 网络预测的结果
        sampling_intervals: 采样协议
        cp_data:血浆

    Returns:预测的18帧图像

    """
    # 由预测的k1-k4 图像和已知的Cp数据生成预测的18帧的数据
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().detach().numpy()  # 128,128,12

    CP_FDG = cp_data
    pred_CT_FDG = update_tracer_concentration_np(reconstruct_for_k_data, CP_FDG, 0)

    f_FDG = np.zeros((128, 128, 18))

    lambda_value = np.log(2) / (20.4 * 60)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FDG[:, :, k] = calculate_xm(start_index, end_index, pred_CT_FDG, lambda_value)
        start_index = end_index + 1

    pred_fdg = f_FDG / np.max(f_FDG)
    # pred_fdg = torch.from_numpy(pred_fdg)
    return pred_fdg


def get_mean_k_data_np(reconstruct_for):
    k1, k2, k3, k4 = np.split(reconstruct_for, 4, axis=1)
    pred_k1 = np.mean(k1.squeeze(), axis=0)
    pred_k2 = np.mean(k2.squeeze(), axis=0)
    pred_k3 = np.mean(k3.squeeze(), axis=0)
    pred_k4 = np.mean(k4.squeeze(), axis=0)
    return pred_k1, pred_k2, pred_k3, pred_k4


def update_tracer_concentration_np(reconstruct_for_k_data, cp_data, number):
    """
    由预测的k1,k2,k3,k4生成预测的TAC曲线
    Args:
        reconstruct_for_k_data:
        cp_data:
        number:

    Returns:

    """
    k1, k2, k3, k4 = get_mean_k_data_np(reconstruct_for_k_data)  # 128x128
    #

    # # 进行最大值最小值归一化
    # if number == 1:
    #     k1 = normalize_array(k1)
    #     k2 = normalize_array(k2)
    #     k3 = normalize_array(k3)
    #     k4 = normalize_array(k4)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    cp_fdg = np.array(cp_data[0].tolist())
    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4
    discriminant = np.maximum(discriminant, 0)  # 将负值替换为零
    alpha1 = (k2 + k3 + k4 - np.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + np.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0
    # 计算 a
    a = np.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])
    # 计算 b
    b = np.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

    # alpha1 = (k2 + k3 + k4 - np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4)) / 2
    # alpha2 = (k2 + k3 + k4 + np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4)) / 2
    # a = k1 * (k3 + k4 - alpha1) / (alpha2 - alpha1)  # a: 128*128
    # b = k1 * (alpha2 - k3 - k4) / (alpha2 - alpha1)

    T = len(cp_fdg)
    array = np.arange(1, T + 1)  # array:(3600,)
    array = array.reshape((1, 1, T))  # 1*1*3600
    a = np.repeat(a[:, :, np.newaxis], T, axis=2)  # a: 128*128*3600
    b = np.repeat(b[:, :, np.newaxis], T, axis=2)  # b: 128*128*3600

    alpha1 = np.repeat(alpha1[:, :, np.newaxis], T, axis=2)
    alpha2 = np.repeat(alpha2[:, :, np.newaxis], T, axis=2)
    part11 = a * cp_fdg  # (128*128*3600)
    part12 = np.exp(-alpha1 * array)  # (128*128*3600)

    part21 = b * cp_fdg  # (128*128*3600)
    part22 = np.exp(-alpha2 * array)  # (128*128*3600)

    # 新卷积方法
    # CT1 = fftconvolve(part11, part12, mode='full', axes=2)
    # CT2 = fftconvolve(part21, part22, mode='full', axes=2)
    # CT1 = CT1[:, :, :T]
    # CT2 = CT2[:, :, :T]

    # CT = CT1 + CT2
    # 新卷积方法
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
    # print(CPET.shape[2] + 1)
    t_values = np.arange(0, CPET.shape[2])  # 假设 CPET 包含3600个时间点，可以自行调整

    time_indices = np.where((t_values >= tms) & (t_values <= tme))[0]  # 获取在 tms 和 tme 范围内的时间索引
    # print(time_indices)
    CPET_sub = CPET[:, :, time_indices]  # 截取对应时间段的 CPET 数据
    t_sub = t_values[time_indices]  # 对应的时间值

    integrand = CPET_sub * np.exp(-lmbda * t_sub)
    xm = trapz(integrand, t_sub)

    return xm


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
