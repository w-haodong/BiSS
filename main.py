import argparse
from operation import test
import warnings
import os
def parse_args():
    parser = argparse.ArgumentParser(description='w-haodong Test For Socliosis - 2025 03')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU')
    parser.add_argument('--work_dir', type=str, default='/CSTemp/whd/work/sco_syb/whd_v2', help='work directory')
    parser.add_argument('--num_epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.5e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=384*3, help='input height')
    parser.add_argument('--input_w', type=int, default=384, help='input width')
    parser.add_argument('--K', type=int, default=17, help='maximum of objects')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--resume', type=str, default='model_final.pth', help='weights_spinal to be resumed')
    parser.add_argument('--data_dir', type=str, default='/CSTemp/whd/datasets/AASCE/wz_pp_data_4/', help='data directory')
    parser.add_argument('--phase', type=str, default='test', help='data directory')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.phase == 'test':
        is_object = test.Network(args)
        is_object.test(args, save=False)
