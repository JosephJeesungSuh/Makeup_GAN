import os
import argparse

from data_loader import load_data
from makeup_gan import MakeupGAN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def train(args):
    data = load_data(args.data_dir, args.batch_size)
    MakeupGAN(data, args).train()


def cli_args_parse():
    parser = argparse.ArgumentParser(
        description='CS280A final project training GAN for makeup transfer'
    )
    # dataset path, intermediate visulation path, model checkpoint save path
    parser.add_argument('--data_dir', type=str, default=os.path.join(ROOT_DIR, 'data'))
    parser.add_argument('--visualization_dir', type=str, default=os.path.join(ROOT_DIR, 'visualization'))
    parser.add_argument('--visualization_interval', type=int, default=30, help='interval in steps')
    parser.add_argument('--model_dir', type=str, default=os.path.join(ROOT_DIR, 'model'))
    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--discrim_update_interval', type=int, default=1, help='how freq. update dis')
    # model hyperparameters
    parser.add_argument('--discrim_repeat_n', type=int, default=3)
    parser.add_argument('--gen_repeat_n', type=int, default=6)
    # loss hyperparameters
    parser.add_argument('--lambda_histogram', type=float, default=0.1)
    parser.add_argument('--lambda_skin', type=float, default=1.0)
    parser.add_argument('--lambda_lip', type=float, default=1.0)
    parser.add_argument('--lambda_eye', type=float, default=1.0)
    parser.add_argument('--lambda_cycle', type=float, default=10)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train(cli_args_parse())