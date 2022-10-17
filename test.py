import os
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy import misc
from torch.utils.data import DataLoader
import torch.nn.functional as F
import apex
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from model.network import Model
from torch.autograd import Variable
from dataset import test_dataset, train_dataset


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def get_loader(train_root, batch_size, trainsize, num_works):
    img_root = os.path.join(train_root, 'Image')
    gt_root = os.path.join(train_root, 'GT_Object')
    gt_edge_root = os.path.join(train_root, 'GT_Edge')
    dataset = train_dataset(img_root, gt_root, gt_edge_root, trainsize)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_works, shuffle=True)
    return train_loader


def adjust_lr(optimizer, step, decay_rate=0.1, decay_step=30):
    decay = decay_rate ** (step // decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        return param_group['lr']


def clip_gradient(optimizer, clip_lr):
    for group in optimizer.param_groups:
        for param in group['param']:
            if param.grad is not None:
                param.grad.data.clamp_(-clip_lr, clip_lr)


def Train():
    global loss, epoch
    parse = argparse.ArgumentParser(description='The training of ERR_Net')
    parse.add_argument('--trainsize', default=352, type=int)
    parse.add_argument('--batch_size', default=6, type=int)
    parse.add_argument('--num_works', default=4, type=int)
    parse.add_argument('--epoch', default=10, type=int)
    parse.add_argument('--lr', default=1e-4, type=float)

    parse.add_argument('--apex', default=False, type=bool)

    parse.add_argument('--backbone_pred', default=True, type=bool)

    parse.add_argument('--start_epoch', default=0, type=int)

    parse.add_argument('--train_root', default='D:\Github\Data\COD10K-v3\Train')
    parse.add_argument('--test_root', default='D:\Github\Data\COD10K-v3\Test')

    parse.add_argument('--save_path', default='./save_path')
    parse.add_argument('--save_epoch', default=10, type=int)

    args = parse.parse_args()

    model = Model(backbone_pre=args.backbone_pred).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 使用apex混合精度训练
    apex.amp.initialize(model, optimizer, opt_level='O1')

    train_loader = get_loader(args.train_root, args.batch_size, args.trainsize, args.num_works)
    total_step = len(train_loader)
    model.train()

    try:
        for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.epoch):
            step_loss = []
            iteration = tqdm(train_loader)
            for step, data_pack in enumerate(iteration):
                optimizer.zero_grad()

                start = time.time()
                img, gt, edge = data_pack
                img = Variable(img).cuda()
                cam_out_g, cam_out_4, cam_out_3, cam_out_2, e_g_out = model(img)
                gt = Variable(gt).cuda()
                edge = Variable(edge).cuda()

                L_c = structure_loss(cam_out_g, gt) + structure_loss(cam_out_4, gt) + structure_loss(cam_out_3, gt) \
                      + structure_loss(cam_out_2, gt)
                L_e = structure_loss(e_g_out, edge)
                loss = L_e + L_c
                step_loss.append(loss.cpu().detach().numpy())

                if args.apex:
                    with apex.amp.scale_loss(loss, optimizer) as scale_loss:
                        scale_loss.backward()
                else:
                    loss.backward()
                end = time.time()
                if step % 10 == 0 or step == total_step:
                    status = (
                        '[time:{:>4f}] => [Epoch Num: {:03d}/{:03d}] => [lr :{:>4f}] => [Global Step: {:04d}/{:04d}] '
                        '=> [Loss_s: {:.4f}]'.format(end - start, epoch, args.epoch, args.lr, step, total_step, loss))
                    iteration.set_description(desc=status)
                optimizer.step()

            if len(step_loss) > 0:
                plt.figure(num=0, figsize=(10, 10))
                plt.plot(step_loss, color='red')
                plt.xlabel('pre_step')
                plt.ylabel('step_loss')
                plt.legend('pre_epoch_step_loss')
                plt.show()

            if os.path.exists(args.save_path) is None:
                os.makedirs(args.save_path, exist_ok=True)

            if epoch % args.save_epoch == 0:
                torch.save(model.state_dict(), args.save_path + 'ERRNet_epoch_%d.pth' % epoch)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model.state_dict(), args.save_path + '\\' + 'ERRNet_epoch_{}.pth'.format(epoch))
        print('Save checkpoints successfully!')
        raise


def Evaluator(
        test_root='./dataset/TestDataset/',
        snap_path='./snapshot/ERRNet_Snapshot.pth',
        save_path='./result/',
        trainsize=352):
    os.makedirs(save_path, exist_ok=True)

    model = Model().cuda()
    model.load_state_dict(torch.load(snap_path))
    model.eval()

    for _data in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']:
        test_data_root = os.path.join(test_root, _data)
        test_dataloader = test_dataset(image_root=test_data_root + 'Imgs/', gt_root=test_data_root + 'GT/',
                                       testsize=trainsize)

        with torch.no_grad():
            for i in range(test_dataloader.size):
                image, gt, name = test_dataloader.load_data()
                gt = np.asarray(gt, np.float32)

                image = image.cuda()

                output = model(image)

                output = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
                output = output.sigmoid().data.cpu().numpy().squeeze()
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)

                misc.imsave(save_path + name, output)
                print('Prediction: {}'.format(save_path + name))


if __name__ == '__main__':
    Train()
    # Evaluator()
