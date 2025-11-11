import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class KMAradar4kmDataset(Dataset):
    def __init__(self, radar_4km_path='/data/KMA/Radar/HSR_4km', year_from=2014, year_to=2021, input_length=7, input_interval=1, output_length=6, output_interval=1):
        self.radar_path = radar_4km_path
        self.year_from = year_from
        self.year_to = year_to
        self.input_length = input_length
        self.input_interval = input_interval
        self.output_length = output_length
        self.output_interval = output_interval
        self.history_length = (input_length-1)*input_interval + output_length*output_interval

        days = [31,28,31,30,31,30,31,31,30,31,30,31]
        times = [f'{hour:02d}{minutes:02d}' for hour in range(0,24) for minutes in range(0,60,10)]
        times.sort(key=lambda x : int(x)) 

        self.radar_list = []
        data_in_years = []
        num_data = 0
        for year in range(year_from, year_to+1):
            for month in range(1,13):
                for day in range(1,days[month-1]+1):
                    for time in times:
                        YYYYMMDD = f"{year:04d}/{month:02d}/{day:02d}"
                        YYYYMMDDHHmm = int(f"{year:04d}{month:02d}{day:02d}{time}")
                        radar_filename = f"{YYYYMMDDHHmm}.tiff"
                        self.radar_list.append(f"{self.radar_path}/{YYYYMMDD}/{radar_filename}")
                        num_data += 1
            data_in_years.append(num_data
                                 )
        self.time_list = list(range(len(self.radar_list)))

        self.history_indices = []
        last_nonexist_index = -1

        for idx, radar_name in enumerate(self.radar_list):
            if idx in data_in_years:
                last_nonexist_index = idx-1

            if os.path.exists(radar_name):
                condition1 = (idx - last_nonexist_index > self.history_length)

                if condition1:
                    self.history_indices.append(idx)
                else:
                    pass
            else:
                last_nonexist_index = idx

    def __len__(self):
        return len(self.history_indices)

    def __getitem__(self, idx):
        final_idx = self.history_indices[idx] 
        current_idx = final_idx - self.output_length*self.output_interval
        init_idx = current_idx - (self.input_length-1)*self.input_interval

        input_radar_history = np.stack([cv2.imread(self.radar_list[time_idx], cv2.IMREAD_UNCHANGED) for time_idx in range(init_idx, current_idx+1, self.input_interval)])
        output_radar_history = np.stack([cv2.imread(self.radar_list[time_idx], cv2.IMREAD_UNCHANGED) for time_idx in range(current_idx+self.output_interval, final_idx+1, self.output_interval)])

        return input_radar_history, output_radar_history



class SEVIRnowcastDataset(Dataset):
    def __init__(self, data_dir, split='train', end=None, pct_validation=0.2, norm=True):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"

        self.file_list = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        self.norm = norm

        total_samples = len(self.file_list) if end is None else min(end, len(self.file_list))
        val_idx = int((1 - pct_validation) * total_samples)

        if split == 'train':
            self.start = 0
            self.end = val_idx
        elif split == 'val':
            self.start = val_idx
            self.end = total_samples
        elif split == 'test':
            self.start = 0
            self.end = total_samples

        self.length = self.end - self.start

    def __len__(self):
        return self.length 
    
    def __getitem__(self, idx):
        file_path = self.file_list[self.start + idx]
        data = np.load(file_path)
        x = torch.tensor(data['IN'], dtype=torch.float32).permute(2, 0, 1)  # (H, W, T) -> (T, H, W)
        y = torch.tensor(data['OUT'], dtype=torch.float32).permute(2, 0, 1)
        
        if self.norm:
            MEAN, SCALE = 33.44, 47.54
            x = (x-MEAN) / SCALE
            y = (y-MEAN) / SCALE

        return x, y


class MeteoNetDataset(Dataset):
    def __init__(self, radar_path='/data/MeteoNet/Radar/SE', year_from=2016, year_to=2018, 
                 input_length=13, input_interval=0.5, output_length=12, output_interval=0.5):

        # input_interval=0.5 means 5 minutes time interval
        # input_interval=1 means 10 minutes time interval
        input_interval = int(input_interval * 2)
        output_interval = int(output_interval * 2)

        self.radar_path = radar_path
        self.year_from = year_from
        self.year_to = year_to
        self.input_length = input_length
        self.input_interval = input_interval
        self.output_length = output_length
        self.output_interval = output_interval
        self.history_length = (input_length-1)*input_interval + output_length*output_interval

        days = [31,28,31,30,31,30,31,31,30,31,30,31]
        times = [f'{hour:02d}{minutes:02d}' for hour in range(0,24) for minutes in range(0,60,5)]
        times.sort(key=lambda x : int(x)) 

        self.radar_list = []
        data_in_years = []
        num_data = 0
        for year in range(year_from, year_to+1):
            for month in range(1,13):
                for day in range(1,days[month-1]+1):
                    day = day + 1 if (year % 4 == 0 and month == 2) else day  # leap year
                    for time in times:
                        YYYYMMDD = f"{year:04d}/{month:02d}/{day:02d}"
                        YYYYMMDDHHmm = int(f"{year:04d}{month:02d}{day:02d}{time}")
                        radar_filename = f"{YYYYMMDDHHmm}.tiff"
                        self.radar_list.append(f"{self.radar_path}/{YYYYMMDD}/{radar_filename}")
                        num_data += 1
            data_in_years.append(num_data)
        self.time_list = list(range(len(self.radar_list)))

        self.history_indices = []
        last_nonexist_index = -1

        for idx, radar_name in enumerate(self.radar_list):
            if idx in data_in_years:
                last_nonexist_index = idx-1

            if os.path.exists(radar_name):
                condition1 = (idx - last_nonexist_index > self.history_length)

                if condition1:
                    self.history_indices.append(idx)
                else:
                    pass
            else:
                last_nonexist_index = idx

    def __len__(self):
        return len(self.history_indices)

    def __getitem__(self, idx):
        final_idx = self.history_indices[idx] 
        current_idx = final_idx - self.output_length*self.output_interval
        init_idx = current_idx - (self.input_length-1)*self.input_interval

        input_radar_history = np.stack([cv2.imread(self.radar_list[time_idx], cv2.IMREAD_UNCHANGED) for time_idx in range(init_idx, current_idx+1, self.input_interval)])
        output_radar_history = np.stack([cv2.imread(self.radar_list[time_idx], cv2.IMREAD_UNCHANGED) for time_idx in range(current_idx+self.output_interval, final_idx+1, self.output_interval)])

        return input_radar_history, output_radar_history