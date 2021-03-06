import argparse

def process_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-j', '--workers', default=36, type=int, metavar='N',
                        help='number of data loading workers')
    # Optimization options
    parser.add_argument('--epochs', default=125, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=7e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoints/imagenet/res18', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture   
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        help='model architecture')
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--block-name', type=str, default='BasicBlock',
                        help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    #Device options
    parser.add_argument('--gpu-id', default='0,1,2,3', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--quan_weight', action='store_true', help='whether use high bit weight.')     
    parser.add_argument('--bit_num', type=int) 
    parser.add_argument('--data', default='path to dataset', type=str)
    parser.add_argument('--warmup_lr_epoch', type=int, default=5)
    parser.add_argument('--N', type=int, default=2)
    parser.add_argument('--M', type=int, default=4)

    args = parser.parse_args()
    return args