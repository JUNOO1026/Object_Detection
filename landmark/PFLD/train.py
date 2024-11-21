import os
import argparse
import logging
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader

from model import PFLDBackbone, AuxiliaryBlock
from dataset import WLFWDataset
from loss import PFLDloss


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))

def train(train_loader, pfld_backbone, auxiliary, criterion, optimizer):
    losses = AverageMeter()

    weighted_loss, loss = None, None
    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        img = img.to(DEVICE)
        landmark_gt = landmark_gt.to(DEVICE)
        attribute_gt = attribute_gt.to(DEVICE)
        euler_angle_gt = euler_angle_gt.to(DEVICE)
        pfld_backbone = pfld_backbone.to(DEVICE)
        auxiliary = auxiliary.to(DEVICE)
        feature, landmarks = pfld_backbone(img)
        euler_angle = auxiliary(feature)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt,
                                        euler_angle_gt, euler_angle, landmarks, args.train_batchsize)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        losses.update(loss.item())

    return weighted_loss, loss


def validate(valid_dataloader, pfld_backbone, auxiliary, criterion):
    pfld_backbone.eval()
    auxiliary.eval()
    losses = []

    with torch.no_grad():
        for img, landmark_gt, attribute_gt, angle_gt in valid_dataloader:
            img = img.to(DEVICE)
            landmark_gt = landmark_gt.to(DEVICE)
            attribute_gt = attribute_gt.to(DEVICE)
            angle_gt = angle_gt.to(DEVICE)
            pfld_backbone = pfld_backbone.to(DEVICE)
            auxiliary = auxiliary.to(DEVICE)
            _, landmark = pfld_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())

    print("====> Evalate:")
    print('Eval set: Average loss: {:.4f}'.format(np.mean(losses)))
    return np.mean(losses)

            

def main(args):
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    pfld_backbone = PFLDBackbone().to(DEVICE)
    auxiliary = AuxiliaryBlock().to(DEVICE)
    criterion = PFLDloss()
    optimizer = torch.optim.Adam([{
        'params': pfld_backbone.parameters()
    },{
        'parmas': auxiliary.parameters()
    }],
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, verbose=True
    )
    if args.resume:
        checkpoint = torch.load(args.resume)
        auxiliary.load_state_dict(checkpoint["auxiliarynet"])
        pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        args.start_epoch = checkpoint["epoch"]

    transform = transform.Compose([transform.ToTensor()])

    wlfwdataset = WLFWDataset(args.dataroot, transform)
    dataloader = DataLoader(wlfwdataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num.workers)
    wlfw_val_dataset = WLFWDataset(args.val_dataroot,transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                    batch_size=args.val_batchsize,
                                    shuffle=False,
                                    num_workers=args.workers)
    _
    # W&B 초기화
    wandb.init(
        project="pfld_pr",  # W&B 프로젝트 이름
        name="pfld_face_landamrk1",      # 실험 이름 (옵션)
        config={                     # 실험 설정 저장 (옵션)
            "learning_rate": args.lr,
            "epochs": args.end_epoch,
            "batch_size": args.batch_size
        }
    )

    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, pfld_backbone,
                                                auxiliary, criterion,
                                                optimizer, epoch)
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint(
            {
                'epoch': epoch,
                'pfld_backbone': pfld_backbone.state_dict(),
                'auxiliarynet': auxiliary.state_dict()
            }, filename)

        val_loss = validate(wlfw_val_dataloader, pfld_backbone, auxiliary,
                            criterion)

        scheduler.step(val_loss)

        # W&B에 기록
        wandb.log({
            'epoch': epoch,
            'weighted_train_loss': weighted_train_loss,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

def parse_args():
    parser = argparse.ArgumentParser(description="PFLD")

    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=500, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/tensorboard",
                        type=str)
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH')

    # --dataset
    parser.add_argument('--dataroot',
                        default='./data/train_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--val_dataroot',
                        default='./data/test_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=256, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

