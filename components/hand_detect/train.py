import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('components/hand_keypoints/')
from utils.model_utils import *
from utils.common_utils import *
from hand_data_iter.datasets import *
from models.resnet import resnet50
from loss.loss import *
import time
import json


def trainer(ops, f_log):
    writer = SummaryWriter(log_dir='logs')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
        if ops.log_flag:
            sys.stdout = f_log
        set_seed(ops.seed)
        # 构建模型
        model = resnet50(pretrained=False, num_classes=ops.num_classes, img_size
        =ops.img_size[0], dropout_factor=ops.dropout)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # 数据集
        dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, \
                                      flag_agu=ops.flag_agu, fix_res=ops.fix_res, vis=False)
        # 加载数据
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)
        # 优化器设计
        optimizer_Adam = torch.optim.Adam(model.parameters(), lr=ops.init_lr, \
                                          betas=(0.9, 0.99), weight_decay=1e-6)
        optimizer = optimizer_Adam
        # 加载 finetune 模型
        if os.access(ops.fintune_model, os.F_OK):  # checkpoint
            chkpt = torch.load(ops.fintune_model, map_location='cpu')
            model.load_state_dict(chkpt)
            del chkpt
        model = model.to(device)
        # 损失函数
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0.  # 损失均值
        loss_idx = 0.  # 损失计算计数器
        flag_change_lr_cnt = 0  # 学习率更新计数器
        init_lr = ops.init_lr  # 学习率

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nEpoch%d#' % epoch)
            model.train()
            # 学习率更新策略
            if loss_mean != 0.:
                if best_loss > (loss_mean / loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean / loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 50:
                        init_lr = init_lr * ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0

            loss_mean = 0.  # 损失均值
            loss_idx = 0.  # 损失计算计数器
            mse_list = []
            for i, (imgs_, tar_) in enumerate(dataloader):
                if use_cuda:
                    imgs_ = imgs_.cuda()  # (batch, channel, height, width)
                    tar_ = tar_.cuda()

                output = model(imgs_.float())
                # 计算当前batch_size的mse
                mse = loss.calc_mes(output.cpu().detach().numpy(), \
                                    tar_.cpu().float().numpy())
                # 将MSE指标写入TensorBoard日志
                writer.add_scalar('MSE', mse, global_step=epoch * len(dataloader) + i)
                mse_list.append(mse)

                if ops.loss_define == 'wing_loss':
                    loss = got_total_loss(output, tar_.float())
                else:
                    loss = criterion(output, tar_.float())
                loss_mean += loss.item()
                loss_idx += 1
                acc = calc_acc(output, tar_.float())
                print('loss: %.4f' % (loss_mean / loss_idx, loss.item()), 'acc : %.4f' % acc)
                # 计算梯度
                loss.backward()
                # 优化器对模型参数更新
                optimizer.step()
                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1
            # 所有batch的mse求平均
            avg_mes = np.mean(mse_list)
            torch.save(model.state_dict(), ops.model_exp + \
                       '{}-size-{}-model_epoch-{}-avg_mse-{}.pth'. \
                       format(ops.model, ops.img_size[0], epoch, avg_mes))
        writer.close()
    except Exception as e:
        print('Exception : ', e)  # 打印异常


if __name__ == "__main__":
    args = parser.parse_args()  # 解析添加参数
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)  # 建立文件夹model_exp
    f_log = None
    unparsed = vars(args)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    fs = open(args.model_exp + 'train_ops.json', "w", encoding='utf-8')  # 读取训练train_ops.json
    json.dump(unparsed, fs, ensure_ascii=False, indent=1)
    fs.close()
    trainer(ops=args, f_log=f_log)  # 模型训练
    if args.log_flag:
        sys.stdout = f_log
    print('well done')
