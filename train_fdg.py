import argparse
import os
import time
import torch.nn.functional as F
import torch
import torch.fft
from scipy.io import loadmat, savemat
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset.pet_dataset_fdg import PetDataset
from torch.optim import lr_scheduler
from model.model_new import MultiOutputReversibleGenerator

from untils.com_untils_train_fdg import compute_mean_loss, compute_mean_loss_rev, calculate_img_patlak3_no_weight, \
    calculate_img_patlak3_no_weight_test, calculate_img_torch

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


BATCH_SIZE = 1  
FRAME = 18  
parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training. ")
parser.add_argument("--weight", type=float, default=1, help="Weight for forward loss. ")
parser.add_argument("--out_path", type=str,
                    default="./exps/ours_fdg_head_sample3_0423_patlak3_no_weight_four_loss/",
                    help="Path to save checkpoint. ")
parser.add_argument("--root1", type=str, default="./data/fdg_zubal_head_sample3_kmin_noise_0423/train",
                    help="train images. ")
parser.add_argument("--root2", type=str, default="./data/fdg_zubal_head_sample3_kmin_noise_0423/test",
                    help="test images. ")
parser.add_argument("--resume", dest='resume', action='store_true', help="Resume training. ")
parser.add_argument("--L", type=int, default=FRAME, help="Batch size for training. ")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

os.makedirs(args.out_path, exist_ok=True)
os.makedirs(args.out_path + "/checkpoint", exist_ok=True)


def main(args):
   
    writer = SummaryWriter(args.out_path)

    
    cp_data = loadmat('./data/fdg_zubal_head_sample3_kmin_noise_0423/CP/CP_FDG.mat')['CP_FDG']
    cp_data = cp_data[:, :3600]
    sampling_intervals = [30, 30, 30, 30, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]

    net = MultiOutputReversibleGenerator(input_channels=24, output_channels=24, num_blocks=8)
   
    net.cuda()

    
    if args.resume:
        net.load_state_dict(torch.load(args.out_path + "/checkpoint/latest.pth"))
        print("[INFO] loaded " + args.out_path + "/checkpoint/latest.pth")

   
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    print("[INFO] Start data loading and preprocessing")

    train_dataset = PetDataset(root_folder=args.root1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  drop_last=True)

    print("[INFO] Start to train")
    step = 0

    for epoch in range(100, 200):

        for i_batch, (
                fdg_batch, fdg_noise_batch, k1_data_batch, k2_data_batch, k3_data_batch, k4_data_batch, ki_data_btach,
                vb_data_batch) in enumerate(
            train_dataloader):  
            step_time = time.time()

            target_forward_fdg = fdg_noise_batch.permute(0, 3, 1, 2).float().cuda()  #
            target_forward_k1_data = k1_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]
            target_forward_k2_data = k2_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]
            target_forward_k3_data = k3_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]
            target_forward_k4_data = k4_data_batch.permute(0, 3, 1, 2).float().cuda()  # [1,12,128,128]

            target_forward_k_data = torch.cat(
                (target_forward_k1_data, target_forward_k2_data, target_forward_k3_data, target_forward_k4_data), dim=1)

            fdg_input = target_forward_fdg[:, 0:12, :, :]
            fdg_input = torch.cat([fdg_input, fdg_input], dim=1) 
            reconstruct_for = net(fdg_input)  
            reconstruct_for = torch.abs(reconstruct_for)
            reconstruct_for = torch.clamp(reconstruct_for, 0, 1)

          
            loss_k1, loss_k2, loss_k3, loss_k4 = compute_mean_loss(reconstruct_for, target_forward_k_data,
                                                                   mse_loss=True)
            forward_loss_k_img = loss_k1 + loss_k2 + loss_k3 + loss_k4
            losses = [loss_k1, loss_k2, loss_k3, loss_k4]
            loss_names = ['loss_k1', 'loss_k2', 'loss_k3', 'loss_k4']
            for loss, name in zip(losses, loss_names):
                writer.add_scalar(name, loss.item(), global_step=step)
           
            pred_ki, pred_vb = calculate_img_patlak3_no_weight(reconstruct_for, target_forward_k_data,
                                                               sampling_intervals,
                                                               cp_data)

            target_ki = ki_data_btach.squeeze().to(dtype=torch.float32).cuda()
            target_vb = vb_data_batch.squeeze().to(dtype=torch.float32).cuda()

            forward_loss_data_patlak = F.huber_loss(pred_ki, target_ki) + F.huber_loss(abs(pred_vb), abs(target_vb))

            writer.add_scalar('forward_loss_data_patlak', forward_loss_data_patlak.item(),
                              global_step=step)

            pred_img = calculate_img_torch(reconstruct_for, sampling_intervals, cp_data)
            forward_img_loss = F.mse_loss(pred_img, fdg_batch.squeeze().float().cuda())
            writer.add_scalar('forward_img_loss', forward_img_loss.item(), global_step=step)

            forward_loss = 1.2 * forward_loss_k_img + forward_loss_data_patlak + forward_img_loss
            writer.add_scalar('forward_loss', forward_loss.item(),
                              global_step=step)

            reconstruct_rev = net(reconstruct_for, rev=True)  
            reconstruct_rev = torch.abs(reconstruct_rev)
            reconstruct_rev = torch.clamp(reconstruct_rev, 0, 1)
           
            rev_loss1, rev_loss2, rev_loss3, rev_loss4 = compute_mean_loss_rev(reconstruct_rev,
                                                                               target_forward_fdg[:, 0:12, :, :],
                                                                               mse_loss=True)
            rev_loss = rev_loss1 + rev_loss2 + rev_loss3 + rev_loss4
            rev_losses = [rev_loss1, rev_loss2, rev_loss3, rev_loss4]
            rev_loss_names = ['rev_loss1', 'rev_loss2', 'rev_loss3', 'rev_loss4']
            for loss, name in zip(rev_losses, rev_loss_names):
                writer.add_scalar(name, loss.item(), global_step=step)
           
            writer.add_scalar('rev_loss', rev_loss.item(), global_step=step)
           
            weight = 1
            loss = weight * forward_loss + rev_loss
            writer.add_scalar('loss', loss.item(),
                              global_step=step)  
           
            optimizer.zero_grad()
            loss.backward()  

            # break
            optimizer.step()
            print(
                "Epoch: %d Step: %d || loss: %.10f rev_loss: %.10f forward_loss: %.10f  forward_loss_data_patlak_loss: "
                "%.10f forward_loss_k_img: %.10f forward_img_loss: %.10f || lr: %f time: %f" % (
                    epoch, step, loss.detach().cpu().numpy(), rev_loss.detach().cpu().numpy(),
                    forward_loss.detach().cpu().numpy(), forward_loss_data_patlak.detach().cpu().numpy(),
                    forward_loss_k_img.detach().cpu().numpy(), forward_img_loss.detach().cpu().numpy(),
                    optimizer.param_groups[0]['lr'], time.time() - step_time
                ))

            step += 1

        if epoch % 1 == 0 or epoch % 999 == 0:
            torch.save(net.state_dict(), args.out_path + "/checkpoint/%04d.pth" % epoch)
            print("[INFO] Successfully saved " + args.out_path + "/checkpoint/%04d.pth" % epoch)
        scheduler.step()


if __name__ == '__main__':
    torch.set_num_threads(4)  
    main(args)
