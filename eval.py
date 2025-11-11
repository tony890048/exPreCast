import numpy as np
import torch
from tqdm import tqdm
from metrics import *
from tabulate import tabulate
import os


def evaluation(dir, step_cnt, model, data_loader, args):

    evaluate_path = f'{args.save_dir}/logs/{dir}'
    os.makedirs(evaluate_path, exist_ok=True)

    if args.dataset == 'KMA':
        threshold = [1, 4, 8, 10, 20, 40, 80]
        time_steps = [int((i+1)*(args.output_interval*10)) for i in range(args.output_length)]
        row_labels = [f'{time_step}m' for time_step in time_steps] + ['Avg']
        zr_converter = ZR()
        mask = torch.from_numpy(np.load('./mask_2023.npy')).to(args.device)
    elif args.dataset == 'SEVIR':
        threshold = [16, 74, 133, 160, 181, 219]
        row_labels = [f'{5*(i+1)}m' for i in range(args.output_length)] + ['Avg']
    elif args.dataset == 'METEONET':
        threshold = [19, 28, 35, 40, 47]
        time_steps = [int((i+1)*(args.output_interval*10)) for i in range(args.output_length)]
        row_labels = [f'{time_step}m' for time_step in time_steps] + ['Avg']

    headers = [f'CSI-{p}' for p in threshold]

    table1 = torch.zeros(3, len(headers)).to(args.device)
    table2 = torch.zeros(args.output_length, len(headers)).to(args.device)
    table3 = torch.zeros(args.output_length, len(headers)).to(args.device)
    table4 = torch.zeros(args.output_length, len(headers)).to(args.device)

    confusion1 = {metric: torch.zeros(3, 3, device=args.device) for metric in headers}
    confusion2 = {metric: torch.zeros(args.output_length, 3, device=args.device) for metric in headers}
    confusion3 = {metric: torch.zeros(args.output_length, 3, device=args.device) for metric in headers}
    confusion4 = {metric: torch.zeros(args.output_length, 3, device=args.device) for metric in headers}

    with torch.no_grad():
        for imgs, gts in tqdm(iter(data_loader)):

            # gts.shape: (B, T, H, W)
            gts = gts.to(args.device)
            imgs = imgs.unsqueeze(1).to(args.device)
            preds = model(imgs)
            preds = preds.squeeze(1)

            if args.dataset == 'KMA':
                gts   = zr_converter.to_rain(100 * gts)   * mask    # rain (mm/h)
                preds = zr_converter.to_rain(100 * preds) * mask    # rain (mm/h)
            elif args.dataset == 'SEVIR':
                MEAN, SCALE = 33.44, 47.54
                gts   = gts * SCALE + MEAN
                preds = preds * SCALE + MEAN
            elif args.dataset == 'METEONET' :
                gts = 70 * gts
                preds = 70 * preds

            for j, key in enumerate(headers):
                confusion1[key][0] += compute_hits_misses_fas(gts, preds, threshold=float(threshold[j]))
                confusion1[key][1] += compute_pooled_confusion(gts, preds, threshold=float(threshold[j]), pool_size=4, mode='max')
                confusion1[key][2] += compute_pooled_confusion(gts, preds, threshold=float(threshold[j]), pool_size=16, mode='max')

                for i in range(args.output_length):
                    gt = gts[:,i].unsqueeze(1)
                    pred = preds[:,i].unsqueeze(1)
                    confusion2[key][i] += compute_hits_misses_fas(gt, pred, threshold=float(threshold[j]))
                    confusion3[key][i] += compute_pooled_confusion(gt, pred, threshold=float(threshold[j]), pool_size=4, mode='max')
                    confusion4[key][i] += compute_pooled_confusion(gt, pred, threshold=float(threshold[j]), pool_size=16, mode='max')


        for j in range(len(headers)):
            table1[0, j] = compute_csi(confusion1[headers[j]][0])
            table1[1, j] = compute_csi(confusion1[headers[j]][1])
            table1[2, j] = compute_csi(confusion1[headers[j]][2])
            for i in range(args.output_length):
                table2[i, j] = compute_csi(confusion2[headers[j]][i])
                table3[i, j] = compute_csi(confusion3[headers[j]][i])
                table4[i, j] = compute_csi(confusion4[headers[j]][i])

        # average over lead time
        table2 = torch.cat((table2, torch.mean(table2, axis=0).unsqueeze(0)), dim=0)
        table3 = torch.cat((table3, torch.mean(table3, axis=0).unsqueeze(0)), dim=0)
        table4 = torch.cat((table4, torch.mean(table4, axis=0).unsqueeze(0)), dim=0)

        # CSI-M: average over threshold
        table1 = torch.cat((table1, torch.mean(table1 ,axis=1).unsqueeze(1)), dim=1)
        table2 = torch.cat((table2, torch.mean(table2 ,axis=1).unsqueeze(1)), dim=1)
        table3 = torch.cat((table3, torch.mean(table3 ,axis=1).unsqueeze(1)), dim=1)
        table4 = torch.cat((table4, torch.mean(table4 ,axis=1).unsqueeze(1)), dim=1)

    with open(f"{evaluate_path}/evaluation_radar_{step_cnt}.log", "a") as eval_log:
        eval_log.write(f"Configuration: {str(args)}\n")
        eval_log.flush()

        eval_log.write('CSI \n')
        eval_log.write(tabulate(table1.detach().cpu().numpy(), headers=headers + ["CSI-M"], showindex=["pool1", "pool4", "pool16"], tablefmt="grid", floatfmt=".4f"))
        eval_log.write('\n')
        eval_log.flush()

        eval_log.write('\n<< Scores at different lead times >>\n')
        eval_log.write('CSI \n')
        eval_log.write(tabulate(table2.detach().cpu().numpy(), headers=headers + ["Avg"], showindex=row_labels, tablefmt="grid", floatfmt=".4f"))
        eval_log.write('\n')
        eval_log.flush()

        eval_log.write('CSI-pool4 \n')
        eval_log.write(tabulate(table3.detach().cpu().numpy(), headers=headers + ["Avg"], showindex=row_labels, tablefmt="grid", floatfmt=".4f"))
        eval_log.write('\n')
        eval_log.flush()

        eval_log.write('CSI-pool16 \n')
        eval_log.write(tabulate(table4.detach().cpu().numpy(), headers=headers + ["Avg"], showindex=row_labels, tablefmt="grid", floatfmt=".4f"))
        eval_log.write('\n')
        eval_log.flush()



def evaluation_others(dir, step_cnt, model, data_loader, args):

    evaluate_path = f'{args.save_dir}/logs/{dir}'
    os.makedirs(evaluate_path, exist_ok=True)

    if args.dataset == 'KMA':
        threshold = [1, 4, 8, 10, 20, 40, 80]
        time_steps = [int((i+1)*(args.output_interval*10)) for i in range(args.output_length)]
        row_labels = [f'{time_step}m' for time_step in time_steps] + ['Avg']
        zr_converter = ZR()
        mask = torch.from_numpy(np.load('./mask_2023.npy')).to(args.device)
    elif args.dataset == 'SEVIR':
        threshold = [16, 74, 133, 160, 181, 219]
        row_labels = [f'{5*(i+1)}m' for i in range(args.output_length)] + ['Avg']
    elif args.dataset == 'METEONET':
        threshold = [19, 28, 35, 40, 47]
        time_steps = [int((i+1)*(args.output_interval*10)) for i in range(args.output_length)]
        row_labels = [f'{time_step}m' for time_step in time_steps] + ['Avg']

    headers2 = [f'HSS-{p}' for p in threshold]

    # for HSS
    table5 = torch.zeros(1, len(headers2)).to(args.device)
    table6 = torch.zeros(args.output_length, len(headers2)).to(args.device)

    confusion1 = {metric: torch.zeros(3, 4, device=args.device) for metric in headers2}
    confusion2 = {metric: torch.zeros(args.output_length, 4, device=args.device) for metric in headers2}
    confusion3 = {metric: torch.zeros(args.output_length, 4, device=args.device) for metric in headers2}
    confusion4 = {metric: torch.zeros(args.output_length, 4, device=args.device) for metric in headers2}

    with torch.no_grad():

        for imgs, gts in tqdm(iter(data_loader)):

            # gts.shape: (B, T, H, W)
            gts = gts.to(args.device)
            imgs = imgs.unsqueeze(1).to(args.device)
            preds = model(imgs)
            preds = preds.squeeze(1)

            if args.dataset == 'KMA':
                gts   = zr_converter.to_rain(100 * gts)   * mask    # rain (mm/h)
                preds = zr_converter.to_rain(100 * preds) * mask    # rain (mm/h)
            elif args.dataset == 'SEVIR':
                MEAN, SCALE = 33.44, 47.54
                gts   = gts * SCALE + MEAN
                preds = preds * SCALE + MEAN
            elif args.dataset == 'METEONET' :
                gts = 70 * gts
                preds = 70 * preds

            # ----- Compute confusion matrix for skill scores: HSS -----
            for j, key in enumerate(headers2):
                confusion1[key][0] += compute_4confusion(gts, preds, threshold=float(threshold[j]))
                confusion1[key][1] += compute_pooled_4confusion(gts, preds, threshold=float(threshold[j]), pool_size=4, mode='max')
                confusion1[key][2] += compute_pooled_4confusion(gts, preds, threshold=float(threshold[j]), pool_size=16, mode='max')

                for i in range(args.output_length):
                    gt = gts[:,i].unsqueeze(1)
                    pred = preds[:,i].unsqueeze(1)
                    confusion2[key][i] += compute_4confusion(gt, pred, threshold=float(threshold[j]))
                    confusion3[key][i] += compute_pooled_4confusion(gt, pred, threshold=float(threshold[j]), pool_size=4, mode='max')
                    confusion4[key][i] += compute_pooled_4confusion(gt, pred, threshold=float(threshold[j]), pool_size=16, mode='max')


        # ---- Compute HSS scores ----
        for j in range(len(headers2)):
            table5[0, j] = compute_hss(confusion1[headers2[j]][0])
            for i in range(args.output_length):
                table6[i, j] = compute_hss(confusion2[headers2[j]][i])

        # HSS average over lead time
        table6 = torch.cat((table6, torch.mean(table6, axis=0).unsqueeze(0)), dim=0)

        # HSS-M: average over threshold
        table5 = torch.cat((table5, torch.mean(table5 ,axis=1).unsqueeze(1)), dim=1)
        table6 = torch.cat((table6, torch.mean(table6 ,axis=1).unsqueeze(1)), dim=1)


    with open(f"{evaluate_path}/evaluation_radar_{step_cnt}.log", "a") as eval_log:
        eval_log.write(f"Configuration: {str(args)}\n")
        eval_log.flush()

        eval_log.write('\nHSS \n')
        eval_log.write(tabulate(table5.detach().cpu().numpy(), headers=headers2 + ["HSS-M"], showindex=["pool1"], tablefmt="grid", floatfmt=".4f"))
        eval_log.write('\n')
        eval_log.flush()

        eval_log.write('\n<< Scores at different lead times >>\n')
        eval_log.write('HSS \n')
        eval_log.write(tabulate(table6.detach().cpu().numpy(), headers=headers2 + ["Avg"], showindex=row_labels, tablefmt="grid", floatfmt=".4f"))
        eval_log.write('\n')
        eval_log.flush()
