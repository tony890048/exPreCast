import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str.upper, choices=['KMA', 'SEVIR', 'METEONET'], default='KMA')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_steps', type=int, default=100_000)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--gpu_id', type=str, default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=6455)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--save_dir', type=str, default='test') 

    args = parser.parse_args()

    if args.dataset == 'KMA':
        args.data_path = '/data/KMA/Radar/HSR_4km'
        args.input_length = 7
        args.input_interval = 1
        args.output_length = 6
        args.output_interval = 1
        args.depths = [2, 6, 2, 2]
        args.frozen_stages = None
        args.upsampling_scale = (1,2,2)
        args.patch_expan_size = (2,4,4)
    elif args.dataset == 'SEVIR':
        args.data_path = '/data/sevir/np'
        args.input_length = 13
        args.output_length = 12
        args.depths = [2, 6, 2, 2]
    elif args.dataset == 'METEONET':
        args.data_path = '/data/MeteoNet/Radar/SE'
        args.input_length = 12
        args.input_interval = 0.5
        args.output_length = 12
        args.output_interval = 0.5
        args.depths = [2, 6, 2, 2]

    args.device_ids = [int(i) for i in args.gpu_id.split(',')]
    args.device = f'cuda:{args.device_ids[0]}'

    return args