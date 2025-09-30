import os
import warnings
from matplotlib import pyplot as plt
import argparse
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from Model_Swin_transformer import swin_tiny_patch4_window7_224 as create_model
from utils_class import train_val_data_split, binary_classifier_train_one_epoch, \
    binary_classifier_evaluate, data_transformer, batch_size_volume, build_optimizer, build_scheduler, Dataset_Loader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")


def main(args):
    global train_loss_sum, train_auc_sum, val_loss_sum, val_auc_sum
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tb_writer = SummaryWriter()
    train_image_path, train_patient_name, train_patient_label, train_patient_follow_up_time, \
        val_image_path, val_patient_name, val_patient_label, val_patient_follow_up_time = \
        train_val_data_split(data_path=args.data_path,
                             seed=args.seed,
                             train_val_cohort_list=args.train_val_cohort_list,
                             Clinical_data=args.Clinical_data,
                             val_rate=args.val_rate)

    train_dataset = Dataset_Loader(image_path=train_image_path,
                                   patient_name=train_patient_name,
                                   patient_label=train_patient_label,
                                   patient_follow_up_time=train_patient_follow_up_time,
                                   transform=data_transformer(split='train'))

    # 实例化验证数据集
    val_dataset = Dataset_Loader(image_path=val_image_path,
                                 patient_name=val_patient_name,
                                 patient_label=val_patient_label,
                                 patient_follow_up_time=val_patient_follow_up_time,
                                 transform=data_transformer(split='val'))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size_volume(volume=batch_size,
                                                                            train_dataset=train_dataset,
                                                                            val_dataset=val_dataset),
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size_volume(volume=batch_size,
                                                                          train_dataset=train_dataset,
                                                                          val_dataset=val_dataset),
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    weights_dict = torch.load(args.weights, map_location=device)
    weights_dict = {key: value for key, value in weights_dict.items() if "head" not in key}
    print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers == 'True':
        for name, para in model.named_parameters():
            para.requires_grad_(False) if "head" not in name else print(f"training {name}")

    parameters = [p for p in model.parameters() if p.requires_grad]
    scaler = torch.cuda.amp.GradScaler()
    final_optimizer = build_optimizer(optimizer=args.optimizer,
                                      params=parameters,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)

    scheduler = build_scheduler(scheduler_name=args.scheduler_name,
                                optimizer=final_optimizer,
                                warm_up_epochs=int(2 * args.epochs / 3),
                                len_train_loader=len(train_loader))

    train_auc_list = []
    val_auc_list = []
    for epoch_num in range(args.epochs):
        train_loss_sum, train_auc_sum = binary_classifier_train_one_epoch(model=model,
                                                                          optimizer=final_optimizer,
                                                                          data_loader=train_loader,
                                                                          device=device,
                                                                          epoch=epoch_num,
                                                                          scaler=scaler,
                                                                          scheduler=scheduler,
                                                                          part='train')

        val_loss_sum, val_auc_sum = binary_classifier_evaluate(model=model,
                                                               data_loader=val_loader,
                                                               device=device,
                                                               epoch=epoch_num,
                                                               part='val  ')

        tags = ["train_loss", "train_auc",
                "val_loss", "val_auc",
                "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss_sum, epoch_num)
        tb_writer.add_scalar(tags[1], train_auc_sum, epoch_num)
        tb_writer.add_scalar(tags[2], val_loss_sum, epoch_num)
        tb_writer.add_scalar(tags[3], val_auc_sum, epoch_num)
        tb_writer.add_scalar(tags[4], final_optimizer.param_groups[0]["lr"], epoch_num)
        train_auc_list.append(train_auc_sum)
        val_auc_list.append(val_auc_sum)

        if train_auc_sum > args.target_train_auc_sum and \
                val_auc_sum > args.target_val_auc_sum:
            torch.save(model.state_dict(),
                       os.path.join(args.weights_save_path,
                                    "{}-{}-seed{}-tr{}-va{}-epoch{}.pth".format(args.section,
                                                                                args.time,
                                                                                args.seed,
                                                                                round(train_auc_sum, 3),
                                                                                round(val_auc_sum, 3),
                                                                                epoch_num)))
            print('---------------------------save-------------------------------')

        print('BestResult\n'
              'tr_auc={:.3f}\n'
              'val_auc={:.3f}\n'
              'epoch={}\n'.format(train_auc_list[val_auc_list.index(max(val_auc_list))],
                                  max(val_auc_list),
                                  val_auc_list.index(max(val_auc_list))))

    if max(train_auc_list) > args.target_train_auc_sum and \
            max(val_auc_list) > args.target_val_auc_sum:
        fig, axs = plt.subplots(1, 2, figsize=(45, 15))
        axs[0].plot(train_auc_list, 'r')
        axs[0].set_title('train,seed={}'.format(args.seed))
        axs[1].plot(val_auc_list, 'r')
        axs[1].set_title('validation,seed={}'.format(args.seed))
        plt.show()
    return train_loss_sum, train_auc_sum, val_loss_sum, val_auc_sum


if __name__ == '__main__':
    lr = 0.0000006
    time = '20250121'
    num_classes = 2
    val_rate = 0.3
    batch_size = 36
    optimizer = 'AdamW'
    weight_decay = 8E-1
    epoch_list = [80]
    section = 'DCE'
    scheduler_name = 'cos'
    seed_list = [1234]
    target_train_auc_sum = 0.5
    target_val_auc_sum = 0.5
    train_val_cohort_list = ['']

    Dir = r'image_path'
    weights_save_path = r'image_path\weight'
    train_weights_path = os.path.join(weights_save_path, 'swin_tiny_patch4_window7_224.pth')
    Clinical_data_path = os.path.join(Dir, "Clinical_ifo.csv")
    Clinical_data = pd.read_csv(Clinical_data_path, index_col=0, encoding='gbk')
    Clinical_data = Clinical_data.iloc[:, :3]
    Clinical_data.columns = ["p_ID", "fustat", "futime"]

    for seed in seed_list:
        for epoch in epoch_list:
            parser = argparse.ArgumentParser()
            parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
            parser.add_argument('--data_path', type=str, default=Dir)
            parser.add_argument('--seed', type=int, default=seed)
            parser.add_argument('--train_val_cohort_list', type=float, default=train_val_cohort_list)
            parser.add_argument('--val_rate', type=float, default=val_rate)
            parser.add_argument('--Clinical_data', type=float, default=Clinical_data)
            parser.add_argument('--batch_size', type=str, default=batch_size)
            parser.add_argument('--num_classes', type=int, default=num_classes)
            parser.add_argument('--weights', type=str, default=train_weights_path, help='initial weights path')
            parser.add_argument('--lr', type=float, default=lr)
            parser.add_argument('--optimizer', type=str, default=optimizer)
            parser.add_argument('--weight_decay', type=float, default=weight_decay)
            parser.add_argument('--freeze_layers', type=bool, default=False)
            parser.add_argument('--scheduler_name', type=str, default=scheduler_name)
            parser.add_argument('--epochs', type=int, default=epoch)
            parser.add_argument('--target_train_auc_sum', type=float, default=target_train_auc_sum)
            parser.add_argument('--target_val_auc_sum', type=float, default=target_val_auc_sum)
            parser.add_argument('--weights_save_path', type=str, default=weights_save_path)
            parser.add_argument('--section', type=str, default=section)
            parser.add_argument('--time', type=str, default=time)
            opt = parser.parse_args()
            train_loss_sum, train_auc_sum, val_loss_sum, val_auc_sum = main(opt)
