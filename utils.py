from dataset import *
from torch.utils.data import Subset

def build_dataset(args):
    if args.dataset == 'KMA':
        train_dataset = KMAradar4kmDataset(
            args.data_path, year_from=2014, year_to=2021,
            input_length=args.input_length, input_interval=args.input_interval,
            output_length=args.output_length, output_interval=args.output_interval
        )
        valid_dataset = KMAradar4kmDataset(
            args.data_path, year_from=2022, year_to=2022,
            input_length=args.input_length, input_interval=args.input_interval,
            output_length=args.output_length, output_interval=args.output_interval
        )
        test_dataset = KMAradar4kmDataset(
            args.data_path, year_from=2023, year_to=2023,
            input_length=args.input_length, input_interval=args.input_interval,
            output_length=args.output_length, output_interval=args.output_interval
        )

    elif args.dataset == 'SEVIR':
        train_dataset = SEVIRnowcastDataset(args.data_path + '/train', split='train')
        valid_dataset = SEVIRnowcastDataset(args.data_path + '/train', split='val')
        test_dataset  = SEVIRnowcastDataset(args.data_path + '/test', split='test')
        
    elif args.dataset == 'METEONET':
        dataset = MeteoNetDataset(args.data_path, year_from=2016, year_to=2018,
                                  input_length=args.input_length, input_interval=args.input_interval,
                                  output_length=args.output_length, output_interval=args.output_interval)
        total_len = len(dataset)
        train_percent = 0.6
        train_len = int(total_len * train_percent)
        valid_len = int(total_len * (1 - train_percent) / 2)
        test_len  = total_len - (train_len + valid_len)

        train_dataset = Subset(dataset, indices=list(range(train_len)))
        valid_dataset = Subset(dataset, indices=list(range(train_len, train_len + valid_len)))
        test_dataset  = Subset(dataset, indices=list(range(train_len + valid_len, total_len)))
    
    return train_dataset, valid_dataset, test_dataset

