import os
import argparse

from data_loader import load_data
from makeup_gan import MakeupGAN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def train(args):
    args.job_id = str(args.job_id)
    if not os.path.exists(args.visualization_dir):
        os.makedirs(args.visualization_dir)
    if not os.path.exists(os.path.join(args.visualization_dir, f'job_id={args.job_id}')):
        os.makedirs(os.path.join(args.visualization_dir, f'job_id={args.job_id}'))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(os.path.join(args.model_dir, f'job_id={args.job_id}')):
        os.makedirs(os.path.join(args.model_dir, f'job_id={args.job_id}'))

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
    parser.add_argument('--model_interval', type=int, default=30, help='interval in steps')
    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--generator_lr', type=float, default=2e-4)
    parser.add_argument('--discriminator_lr', type=float, default=2e-4)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--lr_gamma', type=float, default=0.99, help='per epoch')
    parser.add_argument('--job_id', type=int, default=-1)
    # model hyperparameters
    parser.add_argument('--dis_repeat_n', type=int, default=3)
    parser.add_argument('--gen_repeat_n', type=int, default=6)
    # loss hyperparameters
    parser.add_argument('--lambda_skin', type=float, default=0.1)
    parser.add_argument('--lambda_lip', type=float, default=1.0)
    parser.add_argument('--lambda_eye', type=float, default=1.0)
    parser.add_argument('--lambda_identity', type=float, default=10.0)
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_perceptual', type=float, default=5e-2)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train(cli_args_parse())