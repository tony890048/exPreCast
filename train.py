import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

from config import parse_args
from utils import build_dataset
from model import exPreCast
from loss import *
from optim import WarmupCosineScheduler
from eval import *


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + '/checkpoints', exist_ok=True)
    os.makedirs(args.save_dir + '/logs', exist_ok=True)

    f_log = open(f"{args.save_dir}/logs/train.log", "a")
    f_log.write(f"Configuration: {str(args)}\n")
    f_log.flush()

    train_dataset, valid_dataset, test_dataset = build_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=40, shuffle=False, num_workers=4, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=40, shuffle=False, num_workers=4, drop_last=True)

    model = exPreCast(input_frames=args.input_length, 
                        output_frames=args.output_length,
                        depths=args.depths,
                        upsampling_scale=args.upsampling_scale,
                        patch_expan_size=args.patch_expan_size,
                        frozen_stages=args.frozen_stages).to(args.device)
    f_log.write(f'Number of trainable parameters: {len(model):,}\n')
    f_log.flush()

    model.init_weights()

    model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = WarmupCosineScheduler(optimizer, args.n_steps, base_lr=args.lr).get_scheduler()

    criterion = FACL(args.n_steps).to(args.device)
    step_cnt = 0
    train_loss = 0.
    start_time = time.time()

    while step_cnt < args.n_steps:
        
        for imgs, gts in tqdm(iter(train_loader)):
            optimizer.zero_grad()

            imgs, gts = map(lambda x: x.unsqueeze(1).to(args.device), [imgs, gts])
            preds = model(imgs)

            loss = criterion(preds, gts)

            loss.backward()
            optimizer.step()
            scheduler.step() 
            train_loss += loss.item()

            step_cnt += 1

            if step_cnt % args.log_freq == 0:
                elapsed_time = time.time() - start_time
                f_log.write(f'Step #{step_cnt}: train_loss is {train_loss / args.log_freq:.8f} \t {elapsed_time:.2f} seconds\n')
                f_log.flush()
                train_loss = 0.

            if step_cnt % args.save_freq == 0:
                torch.save(model.state_dict(), f'{args.save_dir}/checkpoints/train_{step_cnt}.pth')

            if step_cnt % args.val_freq == 0:
                model.eval()
                evaluation('validation', step_cnt, model, valid_loader, args)
                evaluation('test', step_cnt, model, test_loader, args)
                model.train()

            if step_cnt == args.n_steps:
                elapsed_time = time.time() - start_time
                f_log.write(f'Final Elapsed Time: {elapsed_time:.2f} seconds\n')
                f_log.flush()
                break

if __name__ == '__main__':
    main()