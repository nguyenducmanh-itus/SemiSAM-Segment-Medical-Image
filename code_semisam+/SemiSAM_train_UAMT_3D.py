import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case

from semisam_plus import semisam_branch
# --- Nhận thông số từ command line ---
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS19', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTS/SemiSAM_UAMT', help='experiment_name')
parser.add_argument('--prompt', type=str,
                    default='unc')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# Tham số cho dữ liệu có nhãn và không nhãn
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=2,
                    help='labeled data')
# Tham số cho hàm mất mát và Mean Teacher
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()

# --- Các hàm hỗ trợ ---
def get_current_consistency_weight(epoch):
    """
    Tính toán trọng số nhất quán hiện tại dựa trên epoch.
    """
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Cập nhật các tham số của mô hình EMA (mô hình giáo viên) bằng cách lấy trung bình trượt theo cấp số nhân
    từ các tham số của mô hình chính (mô hình học sinh).
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
# --- Hàm huấn luyện chính ---
def train(args, snapshot_path):
    """
    Hàm chính để huấn luyện mô hình phân đoạn ảnh y tế bán giám sát.

    Args:
        args: Đối tượng chứa các tham số được phân tích từ dòng lệnh.
        snapshot_path: Đường dẫn thư mục để lưu các checkpoint mô hình và nhật ký.
    """
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model(ema=False):
        """
        Hàm nội bộ để tạo ra một thể hiện của mô hình mạng.
        """
        # Định nghĩa mạng
        net = net_factory_3d(net_type=args.model,
                             in_chns=1, class_num=num_classes)
        model = net.cuda()# Chuyển mô hình lên GPU
        if ema:# Nếu là mô hình EMA (giáo viên), không tính gradient cho nó
            for param in model.parameters():
                param.detach_()# Ngắt kết nối tham số khỏi đồ thị tính toán
        return model

    model = create_model()# Tạo mô hình học sinh (student model)
    ema_model = create_model(ema=True)# Tạo mô hình giáo viên (teacher model - EMA model)

    # Khởi tạo tập dữ liệu huấn luyện

    db_train = BraTS2019(base_dir=train_data_path,
                         split='train', # Sử dụng tập huấn luyện
                         num=None,# Tải tất cả các mẫu trong tập huấn luyện
                         transform=transforms.Compose([# Chuỗi các phép biến đổi dữ liệu
                             RandomRotFlip(),# Xoay hoặc lật ngẫu nhiên
                             RandomCrop(args.patch_size),# Cắt ngẫu nhiên một patch với kích thước đã định
                             ToTensor(),# Chuyển đổi dữ liệu sang Tensor PyTorch
                         ]))

    
    # Xác định chỉ mục cho dữ liệu có nhãn và không nhãn
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 250))
    # Tạo batch sampler để lấy dữ liệu từ cả hai nhóm (có nhãn và không nhãn)
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    # Khởi tạo DataLoader
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()# Đặt mô hình học sinh vào chế độ huấn luyện
    ema_model.train()# Đặt mô hình giáo viên vào chế độ huấn luyện 
    # Khởi tạo bộ tối ưu hóa và các hàm mất mát
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0# Biến đếm số lần lặp tổng cộng
    max_epoch = max_iterations // len(trainloader) + 1# Tính số epoch tối đa
    best_performance = 0.0# Lưu trữ hiệu suất tốt nhất đạt được trên tập thẩm định
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # Tách riêng batch dữ liệu không được gán nhãn
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            # Thêm nhiễu ngẫu nhiên vào dữ liệu không được gán nhãn cho mô hình EMA (tăng cường mạnh mẽ)
            # Đây là một dạng nhiễu Gaussian được cắt bớt
            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            # --- Forward pass ---
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            T = 8
            _, _, d, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, d, w, h]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + \
                    torch.clamp(torch.randn_like(
                        volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *
                          (i + 1)] = ema_model(ema_inputs)
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, d, w, h)
            preds = torch.mean(preds, dim=0)
            # Tính toán độ không chắc chắn bằng Entropy của các dự đoán
            # Entropy cao = độ không chắc chắn cao
            uncertainty = -1.0 * \
                torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
            # --- Tính toán Loss ---
            # Loss có giám sát (chỉ trên dữ liệu được gán nhãn)
            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs])
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            # Loss nhất quán cho Mean Teacher CÓ GIÁM SÁT ĐỘ KHÔNG CHẮC CHẮN (UAMT)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = losses.softmax_mse_loss(
                outputs[args.labeled_bs:], ema_output)  # (batch, 2, 112,112,80)
            # Ngưỡng (threshold) cho độ không chắc chắn. Ngưỡng này tăng dần theo thời gian huấn luyện.
            # (0.75 + 0.25 * sigmoid_rampup) * log(2) là công thức để kiểm soát ngưỡng
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num,
                                                        max_iterations))*np.log(2)
            # Tạo mask dựa trên ngưỡng độ không chắc chắn:
            # Các vùng có entropy (uncertainty) thấp hơn ngưỡng sẽ được chọn (mask = 1.0)
            # Các vùng có entropy cao hơn ngưỡng sẽ bị bỏ qua (mask = 0.0)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(
                mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
             # --- Nhánh Semi-SAM ---
            # samseg_mask: mask phân đoạn từ SAM (có thể là SAM-Med3D)
            # uncsam: bản đồ độ không chắc chắn từ SAM
            samseg_mask, uncsam = semisam_branch(volume_batch, outputs_soft[:,0:1,:,:,:], generalist='SAM-Med3D',prompt=args.prompt)
            # Chuyển mask từ SAM sang định dạng xác suất mềm (2 kênh: nền và đối tượng)
            samseg_soft = torch.cat((1 - samseg_mask, samseg_mask), dim=1)
             # Loss Dice từ SAM (so sánh đầu ra SAM với nhãn thật, chỉ dùng cho dữ liệu có nhãn)
            # Dòng này có thể dùng để đánh giá SAM hoặc làm một thành phần loss phụ
            sam_dice = (label_batch[:args.labeled_bs] - samseg_soft[:args.labeled_bs] ) **2

            # Tính toán loss nhất quán cho SAM
            if args.prompt == 'unc':
                # Trọng số của consistency được điều chỉnh bởi độ không chắc chắn (uncsam)
                # Tập trung vào các vùng mà SAM không chắc chắn
                sam_consistency_dist = (outputs_soft[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2
                sam_consistency = torch.mean(
                    sam_consistency_dist * uncsam) / (torch.mean(uncsam) + 1e-8) + torch.mean(uncsam)
            else:# Nếu prompt không phải 'unc'
                #MSE giữa đầu ra của mô hình và đầu ra của SAM
                sam_consistency = torch.mean(
                    (outputs_soft[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2)


            consistency_weight_sam = get_current_consistency_weight((args.max_iterations - iter_num )//150)
            # Chuẩn hóa sam_consistency (đảm bảo giá trị nằm trong khoảng hợp lý)
            sam_consistency = torch.sum(sam_consistency)/(torch.sum(sam_consistency)+1e-16) 
            # Loss nhất quán từ SAM
            sam_con_loss = 0.1 * consistency_weight_sam * sam_consistency

            # Tổng Loss cuối cùng

            loss = supervised_loss + consistency_weight * consistency_loss + sam_con_loss
             # --- Backpropagation và cập nhật mô hình ---
            optimizer.zero_grad()# Đặt gradient về 0
            loss.backward()# Tính toán gradient
            optimizer.step()# Cập nhật tham số mô hình học sinh
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)# Cập nhật tham số mô hình giáo viên (EMA)
            # --- Điều chỉnh Learning Rate ---
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # Hàm giảm LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # --- Ghi nhật ký TensorBoard ---
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            # --- Trực quan hóa kết quả (cứ sau 20 lần lặp) ---
            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)
            # --- Đánh giá và lưu mô hình (cứ sau 200 lần lặp) ---
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()
             # --- Lưu checkpoint mô hình (cứ sau 200 lần lặp) ---
            if iter_num % 200 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            # --- Điều kiện dừng huấn luyện ---
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

# --- Khối thực thi chính ---
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
