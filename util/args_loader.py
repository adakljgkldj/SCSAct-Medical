import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    parser.add_argument('--in-dataset', default="NCT", type=str,help='CIFAR-100 imagenet HAM10000 lung')
    parser.add_argument('--out-datasets',default=['dtd', 'inat', 'places50', 'sun50', 'bnz', 'nkj', 'njcell', 'fallmud', 'NCT2', 'rx','rxcs'], type=list,help="['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd']")
    parser.add_argument('--name', default="mobilenet", type=str, help='neural network name and training set')
    parser.add_argument('--model-arch', default='resnet', type=str,help='model architecture [resnet50]')


    parser.add_argument('--threshold_h', default=1.2651828408241281, type=float,help='sparsity level')
    parser.add_argument('--threshold_l', default=0.010410694405436518, type=float, help='sparsity level')

    parser.add_argument('--a', default=0.8, type=float, help='similiraity and variance')
    parser.add_argument('--k', default=500, type=int, help='number of selected channels')

    parser.add_argument('--method', default='msp', type=str, help='odin mahalanobis CE_with_Logst,msp_bats')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')

    parser.add_argument('--gpu', default='1', type=str, help='gpu index')
    parser.add_argument('-b', '--batch-size', default=25, type=int, help='mini-batch size')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.add_argument('--m', default=0)
    parser.add_argument('--n', default=0)

    parser.set_defaults(argument=True)
    args = parser.parse_args()
    return args
