import argparse
from glob import glob

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FoodKT
from models import ImageModel
from utils import *


@torch.no_grad()
def test(args, model, val_loader):
    model.eval()
    batch_iter = tqdm(enumerate(val_loader), 'Validating', total=len(val_loader), ncols=120)

    preds, answer = [], []
    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        label = batch_item['label'].to(args.device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img)
        preds.extend(torch.argmax(pred, dim=1).clone().cpu().numpy())
        answer.extend(label.cpu().numpy())

    preds = np.array([label_decoder[int(val)] for val in preds])
    answer = np.array([label_decoder[int(val)] for val in answer])
    confusion_matrix = pd.crosstab(answer, preds, rownames=['answer'], colnames=['preds'])
    confusion_matrix.to_csv(f'{args.save_dir}/{args.csv_name}.csv', index=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, default='/home/work/team01/food-kt/data')
    parser.add_argument('-vdd', '--val_data_dir', type=str, default='/home/work/team01/food-kt/data/val')
    parser.add_argument('-sd', '--save_dir', type=str, default='/home/work/team01/food-kt/submissions')
    parser.add_argument('-cv', '--csv_name', type=str, default='confusion_mat')
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                        default='/home/work/team01/food-kt/ckpt/tf_efficientnet_b4_ns_0927_062809/ckpt_best.pt')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-is', '--img_size', type=int, default=384)
    parser.add_argument('-nw', '--num_workers', type=int, default=4)
    parser.add_argument('-m', '--model', type=str, default='tf_efficientnet_b4_ns')
    parser.add_argument('-av', '--aug_ver', type=int, default=0)
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('-se', '--seed', type=int, default=42)

    # data split configs:
    parser.add_argument('-ds', '--data_split', type=str, default='StratifiedKFold',
                        choices=['Split_base', 'StratifiedKFold'])
    parser.add_argument('-ns', '--n_splits', type=int, default=5)
    parser.add_argument('-vr', '--val_ratio', type=float, default=0.2)

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #### SEED EVERYTHING ####
    seed_everything(args.seed)
    #########################
    
    #### SET DATASET ####
    label_description = sorted(os.listdir(os.path.join(args.data_dir, 'train')))
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}

    val_data = sorted(glob(f'{args.val_data_dir}/*/*.jpg'))
    val_label = [data.split('/')[-2] for data in val_data]    # '가자미전'
    val_labels = [label_encoder[k] for k in val_label]        # 0
    #####################

    #### LOAD DATASET ####
    val_dataset = FoodKT(args, val_data, labels=val_labels, mode='valid')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    print('> DATAMODULE BUILT')
    ######################

    #### LOAD MODEL ####
    model = ImageModel(model_name=args.model, class_n=len(label_description), mode='valid')
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print('> MODEL BUILT')
    ####################
    
    #### INFERENCE START ####
    print('> START INFERENCE ')
    test(args, model, val_loader)
    print('> SAVE CONFUSION MATRIX')
    #########################