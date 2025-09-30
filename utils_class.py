import math
import os
import random
import sys

import SimpleITK
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class Dataset_Loader(Dataset):
    def __init__(self,
                 image_path: list,
                 patient_name: list,
                 patient_label: list,
                 patient_follow_up_time: list,
                 transform=None):
        self.image_path = image_path
        self.patient_name = patient_name
        self.patient_label = patient_label
        self.patient_follow_up_time = patient_follow_up_time
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image_path = self.image_path[item]
        image = SimpleITK.ReadImage(image_path)
        image = SimpleITK.GetArrayFromImage(image)
        image = image.transpose(2, 1, 0)
        image = image.astype(np.float32)
        image = self.transform(image)
        patient_name = self.patient_name[item]
        patient_label = self.patient_label[item]
        patient_follow_up_time = self.patient_follow_up_time[item]
        return image, image_path, patient_name, patient_label, patient_follow_up_time

    @staticmethod
    def collate_fn(batch):
        images, image_paths, patient_names, patient_labels, patient_follow_up_times = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        patient_labels = torch.as_tensor(patient_labels)
        patient_follow_up_times = torch.as_tensor(patient_follow_up_times)
        return images, image_paths, patient_names, patient_labels, patient_follow_up_times


class C_index(nn.Module):
    def __init__(self):
        super(C_index, self).__init__()

    def forward(self, risk_score, follow_up_time, label):
        from lifelines.utils import concordance_index
        result = concordance_index(follow_up_time, -risk_score, label)
        return result


class Regularization(object):
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class cox_loss(nn.Module):
    def __init__(self):
        super(cox_loss, self).__init__()
        self.reg = Regularization(order=2, weight_decay=0.1)

    def forward(self, risk_pred, durations, events):
        eps = 1e-7
        idx = durations.sort(descending=True)[1]
        events = events[idx]
        risk_pred = risk_pred[idx]

        if events.dtype is torch.bool:
            events = events.float()
        events = events.view(-1)
        risk_pred = risk_pred.view(-1)
        gamma = risk_pred.max()
        log_sum_h = risk_pred.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        loss = - risk_pred.sub(log_sum_h).mul(events).sum().div(events.sum())
        return loss


def train_val_data_split(data_path, seed, train_val_cohort_list, Clinical_data, val_rate):
    random.seed(seed)
    assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

    train_image_path = []
    train_patient_name = []
    train_patient_label = []
    train_patient_follow_up_time = []

    val_image_path = []
    val_patient_name = []
    val_patient_label = []
    val_patient_follow_up_time = []

    patient_name_list = []
    for single_patient_name in Clinical_data["p_ID"]:
        patient_name_list.append(single_patient_name)
    patient_name_list = list(set(patient_name_list))
    patient_name_list.sort()
    train_patient_names = random.sample(patient_name_list, k=int(len(patient_name_list) * (1 - val_rate)))
    internal_val_patient_names = [item for item in patient_name_list if item not in train_patient_names]

    for train_val_cohort in train_val_cohort_list:
        single_cohort_path = os.path.join(data_path, train_val_cohort)
        single_cohort_image_list = [os.path.join(single_cohort_path, i) for i in os.listdir(single_cohort_path)]
        for image_path in single_cohort_image_list:
            patient_pathname = image_path.split('\\')[-1]
            if True:
                for single_patient_name in internal_val_patient_names:
                    if str(single_patient_name) in str(patient_pathname):
                        val_image_path.append(os.path.join(image_path))
                        val_patient_name.append(Clinical_data.loc[patient_pathname, 'p_ID'])
                        val_patient_label.append(Clinical_data.loc[patient_pathname, 'label'])
                        val_patient_follow_up_time.append(Clinical_data.loc[patient_pathname, 'follow_up_time'])

                for single_patient_name in train_patient_names:
                    if str(single_patient_name) in str(patient_pathname):
                        train_image_path.append(os.path.join(image_path))
                        train_patient_name.append(Clinical_data.loc[patient_pathname, 'p_ID'])
                        train_patient_label.append(Clinical_data.loc[patient_pathname, 'label'])
                        train_patient_follow_up_time.append(Clinical_data.loc[patient_pathname, 'follow_up_time'])

                        if Clinical_data.loc[patient_pathname, 'label'] == 1:
                            train_image_path.append(os.path.join(image_path))
                            train_patient_name.append(Clinical_data.loc[patient_pathname, 'p_ID'])
                            train_patient_label.append(Clinical_data.loc[patient_pathname, 'label'])
                            train_patient_follow_up_time.append(Clinical_data.loc[patient_pathname, 'follow_up_time'])

    print("\n{} images for training.".format(len(train_image_path)))
    print("{} images for validation.".format(len(val_image_path)))
    assert len(train_patient_names) > 0, "number of training images must greater than 0."
    assert len(internal_val_patient_names) > 0, "number of validation images must greater than 0."

    train_data = pd.DataFrame({'image_path': train_image_path,
                               'p_ID': train_patient_name,
                               'label': train_patient_label,
                               'follow_up_time': train_patient_follow_up_time})
    train_data.to_csv(data_path + "\\seed" + str(seed) + "_train_dataset.csv", encoding='gbk', index=False)

    val_data = pd.DataFrame({'image_path': val_image_path,
                             'p_ID': val_patient_name,
                             'label': val_patient_label,
                             'follow_up_time': val_patient_follow_up_time})
    val_data.to_csv(data_path + "\\seed" + str(seed) + "_internal_dataset.csv", encoding='gbk', index=False)

    return train_image_path, train_patient_name, train_patient_label, train_patient_follow_up_time, \
        val_image_path, val_patient_name, val_patient_label, val_patient_follow_up_time


def make_test_data(data_path, test_cohort_list, Clinical_data):
    test_image_path = []
    test_patient_name = []
    test_patient_label = []
    test_patient_follow_up_time = []

    for test_cohort in test_cohort_list:
        single_cohort_path = os.path.join(data_path, test_cohort)
        single_cohort_image_list = [os.path.join(single_cohort_path, i) for i in os.listdir(single_cohort_path)]
        for image_path in single_cohort_image_list:
            if True:
                patient_pathname = image_path.split('\\')[-1]
                test_image_path.append(os.path.join(image_path))
                test_patient_name.append(Clinical_data.loc[patient_pathname, "p_ID"])
                test_patient_label.append(Clinical_data.loc[patient_pathname, "label"])
                test_patient_follow_up_time.append(Clinical_data.loc[patient_pathname, "follow_up_time"])

    print("{} images for testing.".format(len(test_image_path)))
    assert len(test_image_path) > 0, "number of testning images must greater than 0."

    test_data = pd.DataFrame({'image_path': test_image_path,
                              'p_ID': test_patient_name,
                              'label': test_patient_label,
                              'follow_up_time': test_patient_follow_up_time})
    test_data.to_csv(data_path + "\\" + "test_dataset.csv", encoding='gbk', index=False)

    return test_image_path, test_patient_name, test_patient_label, test_patient_follow_up_time


def cox_train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, scheduler, part):
    model.train()
    data_loader = tqdm(data_loader, file=sys.stdout)
    optimizer.zero_grad()
    Cindex_Function = C_index()
    sum_c_index = torch.zeros(1).to(device)
    criterion = cox_loss()
    sum_loss = torch.zeros(1).to(device)
    predict_data = pd.DataFrame(columns=["p_ID", "label", "follow_up_time", "image_paths", "risk_probability"])
    for step, data in enumerate(data_loader):
        images, image_paths, patient_names, patient_labels, patient_follow_up_times = data
        with autocast():
            risk_probability = model(images.to(device))
        predict_single_data = pd.DataFrame({"p_ID": patient_names,
                                            "label": patient_labels,
                                            "follow_up_time": patient_follow_up_times,
                                            "image_paths": image_paths,
                                            "risk_probability": risk_probability[:, 0].cpu().detach().numpy()
                                            })
        predict_data = predict_data._append(predict_single_data)

        risk_score = torch.tensor(predict_data['risk_probability'].astype(float).values, dtype=torch.float32)
        label = torch.tensor(predict_data['label'].astype(float).values, dtype=torch.long)
        follow_up_time = torch.tensor(predict_data['follow_up_time'].astype(float).values, dtype=torch.float32)

        sum_c_index = Cindex_Function(risk_score, follow_up_time, label)
        loss = criterion(risk_probability, patient_follow_up_times.to(device), patient_labels.to(device))
        sum_loss += loss
        scaler.scale(loss).backward()
        data_loader.desc = "[{} epoch {}] [C_index: {:.3f}] [loss:{:.3f}]".format(part,
                                                                                  epoch,
                                                                                  sum_c_index,
                                                                                  sum_loss.item() / (step + 1))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    return sum_loss.item(), sum_c_index.item()


def binary_classifier_train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, scheduler, part):
    model.train()
    data_loader = tqdm(data_loader, file=sys.stdout)
    optimizer.zero_grad()
    sum_auc = torch.zeros(1).to(device)
    criterion = nn.CrossEntropyLoss()
    sum_loss = torch.zeros(1).to(device)
    predict_data = pd.DataFrame(columns=["p_ID", "label", "follow_up_time", "image_paths", "type_one", "type_two"])
    for step, data in enumerate(data_loader):
        images, image_paths, patient_names, patient_labels, patient_follow_up_times = data
        with autocast():
            risk_probability = model(images.to(device))
            risk_probability_else = np.exp(risk_probability.cpu().detach().numpy())
        predict_single_data = pd.DataFrame({"p_ID": patient_names,
                                            "label": patient_labels,
                                            "follow_up_time": patient_follow_up_times,
                                            "image_paths": image_paths,
                                            "type_one": risk_probability_else[:, 0] / risk_probability_else.sum(axis=1),
                                            "type_two": risk_probability_else[:, 1] / risk_probability_else.sum(axis=1)
                                            })
        predict_data = predict_data._append(predict_single_data)
        type_two_score = torch.tensor(predict_data['type_two'].astype(float).values, dtype=torch.float32)
        label = torch.tensor(predict_data['label'].astype(float).values, dtype=torch.long)

        sum_auc = roc_auc_score(label, type_two_score)
        loss = criterion(risk_probability.to(device), patient_labels.to(device))
        sum_loss += loss
        scaler.scale(loss).backward()
        data_loader.desc = "[{} epoch {}] [AUC: {:.3f}] [loss:{:.3f}]".format(part,
                                                                              epoch,
                                                                              sum_auc,
                                                                              sum_loss.item() / (step + 1))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    return sum_loss.item(), sum_auc.item()


@torch.no_grad()
def cox_evaluate(model, data_loader, device, epoch, part):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    Cindex_Function = C_index()
    sum_c_index = torch.zeros(1).to(device)
    criterion = cox_loss()
    sum_loss = torch.zeros(1).to(device)
    predict_data = pd.DataFrame(columns=["p_ID", "label", "follow_up_time", "image_paths", "risk_probability"])
    for step, data in enumerate(data_loader):
        images, image_paths, patient_names, patient_labels, patient_follow_up_times = data
        with autocast():
            risk_probability = model(images.to(device))
        predict_single_data = pd.DataFrame({"p_ID": patient_names,
                                            "label": patient_labels,
                                            "follow_up_time": patient_follow_up_times,
                                            "image_paths": image_paths,
                                            "risk_probability": risk_probability[:, 0].cpu().detach().numpy()
                                            })
        predict_data = predict_data._append(predict_single_data)

        risk_score = torch.tensor(predict_data['risk_probability'].astype(float).values, dtype=torch.float32)
        label = torch.tensor(predict_data['label'].astype(float).values, dtype=torch.long)
        follow_up_time = torch.tensor(predict_data['follow_up_time'].astype(float).values, dtype=torch.float32)

        sum_c_index = Cindex_Function(risk_score, follow_up_time, label)
        loss = criterion(risk_probability, patient_follow_up_times.to(device), patient_labels.to(device))
        sum_loss += loss
        data_loader.desc = "[{} epoch {}] [C_index: {:.3f}] [loss:{:.3f}]".format(part,
                                                                                  epoch,
                                                                                  sum_c_index,
                                                                                  sum_loss.item() / (step + 1))
    return sum_loss.item(), sum_c_index.item()


@torch.no_grad()
def binary_classifier_evaluate(model, data_loader, device, epoch, part):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    sum_auc = torch.zeros(1).to(device)
    criterion = nn.CrossEntropyLoss()
    sum_loss = torch.zeros(1).to(device)
    predict_data = pd.DataFrame(columns=["p_ID", "label", "follow_up_time", "image_paths", "type_one", "type_two"])
    for step, data in enumerate(data_loader):
        images, image_paths, patient_names, patient_labels, patient_follow_up_times = data
        with autocast():
            risk_probability = model(images.to(device))
            risk_probability_else = np.exp(risk_probability.cpu().detach().numpy())
        predict_single_data = pd.DataFrame({"p_ID": patient_names,
                                            "label": patient_labels,
                                            "follow_up_time": patient_follow_up_times,
                                            "image_paths": image_paths,
                                            "type_one": risk_probability_else[:, 0] / risk_probability_else.sum(axis=1),
                                            "type_two": risk_probability_else[:, 1] / risk_probability_else.sum(axis=1)
                                            })
        predict_data = predict_data._append(predict_single_data)
        type_two_score = torch.tensor(predict_data['type_two'].astype(float).values, dtype=torch.float32)
        label = torch.tensor(predict_data['label'].astype(float).values, dtype=torch.long)

        sum_auc = roc_auc_score(label, type_two_score)
        loss = criterion(risk_probability.to(device), patient_labels.to(device))
        sum_loss += loss
        data_loader.desc = "[{} epoch {}] [AUC: {:.3f}] [loss:{:.3f}]".format(part,
                                                                              epoch,
                                                                              sum_auc,
                                                                              sum_loss.item() / (step + 1))
    return sum_loss.item(), sum_auc.item()


@torch.no_grad()
def cox_predict(model, data_loader, device, epoch, part):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    Cindex_Function = C_index()
    sum_c_index = torch.zeros(1).to(device)
    criterion = cox_loss()
    sum_loss = torch.zeros(1).to(device)
    predict_data = pd.DataFrame(columns=["p_ID", "label", "follow_up_time", "image_paths", "risk_probability"])
    for step, data in enumerate(data_loader):
        images, image_paths, patient_names, patient_labels, patient_follow_up_times = data
        with autocast():
            risk_probability = model(images.to(device))
        predict_single_data = pd.DataFrame({"p_ID": patient_names,
                                            "label": patient_labels,
                                            "follow_up_time": patient_follow_up_times,
                                            "image_paths": image_paths,
                                            "risk_probability": risk_probability[:, 0].cpu().detach().numpy()
                                            })
        predict_data = predict_data._append(predict_single_data)

        risk_score = torch.tensor(predict_data['risk_probability'].astype(float).values, dtype=torch.float32)
        label = torch.tensor(predict_data['label'].astype(float).values, dtype=torch.long)
        follow_up_time = torch.tensor(predict_data['follow_up_time'].astype(float).values, dtype=torch.float32)

        sum_c_index = Cindex_Function(risk_score, follow_up_time, label)
        loss = criterion(risk_probability, patient_follow_up_times.to(device), patient_labels.to(device))
        sum_loss += loss
        data_loader.desc = "[{} epoch {}] [C_index: {:.3f}] [loss:{:.3f}]".format(part,
                                                                                  epoch,
                                                                                  sum_c_index,
                                                                                  sum_loss.item() / (step + 1))
    return sum_loss.item(), sum_c_index.item(), predict_data


@torch.no_grad()
def binary_classifier_predict(model, data_loader, device, epoch, part):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    sum_auc = torch.zeros(1).to(device)
    criterion = nn.CrossEntropyLoss()
    sum_loss = torch.zeros(1).to(device)
    predict_data = pd.DataFrame(columns=["p_ID", "label", "follow_up_time", "image_paths", "type_one", "type_two"])
    for step, data in enumerate(data_loader):
        images, image_paths, patient_names, patient_labels, patient_follow_up_times = data
        with autocast():
            risk_probability = model(images.to(device))
            risk_probability_else = np.exp(risk_probability.cpu().detach().numpy())
        predict_single_data = pd.DataFrame({"p_ID": patient_names,
                                            "label": patient_labels,
                                            "follow_up_time": patient_follow_up_times,
                                            "image_paths": image_paths,
                                            "type_one": risk_probability_else[:, 0] / risk_probability_else.sum(axis=1),
                                            "type_two": risk_probability_else[:, 1] / risk_probability_else.sum(axis=1)
                                            })
        predict_data = predict_data._append(predict_single_data)
        type_two_score = torch.tensor(predict_data['type_two'].astype(float).values, dtype=torch.float32)
        label = torch.tensor(predict_data['label'].astype(float).values, dtype=torch.long)

        sum_auc = roc_auc_score(label, type_two_score)
        loss = criterion(risk_probability.to(device), patient_labels.to(device))
        sum_loss += loss
        data_loader.desc = "[{} epoch {}] [AUC: {:.3f}] [loss:{:.3f}]".format(part,
                                                                              epoch,
                                                                              sum_auc,
                                                                              sum_loss.item() / (step + 1))
    return sum_loss.item(), sum_auc.item(), predict_data


def build_optimizer(optimizer, params, lr, weight_decay):
    # define optimizers
    if optimizer == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=0, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999),
        )
    elif optimizer == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay
        )


def build_scheduler(scheduler_name, optimizer, warm_up_epochs, len_train_loader):
    if scheduler_name == "warmup":
        def create_lr_scheduler(optimizer_name,
                                num_step: int,
                                epochs: int,
                                warmup=True,
                                warmup_epochs=warm_up_epochs,
                                warmup_factor=1e-3,
                                end_factor=1e-6):
            assert num_step > 0 and epochs > 0
            if warmup is False:
                warmup_epochs = 0

            def f(x):
                if warmup is True and x <= (warmup_epochs * num_step):
                    alpha = float(x) / (warmup_epochs * num_step)
                    return warmup_factor * (1 - alpha) + alpha
                else:
                    current_step = (x - warmup_epochs * num_step)
                    cosine_steps = (epochs - warmup_epochs) * num_step
                    return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

            return torch.optim.lr_scheduler.LambdaLR(optimizer_name, lr_lambda=f)

        return create_lr_scheduler(optimizer_name=optimizer,
                                   num_step=len_train_loader,
                                   warmup=True,
                                   warmup_epochs=warm_up_epochs)

    elif scheduler_name == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)


def data_transformer(split):
    # t, p = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    t, p = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if split == 'train':
        train_data_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((224, 224)),
                                                   # transforms.RandomRotation(230),
                                                   # transforms.RandomHorizontalFlip(p=0.5),
                                                   # transforms.RandomVerticalFlip(p=0.5),
                                                   # transforms.ColorJitter(brightness=0.25, contrast=0.25),
                                                   transforms.Normalize(t, p)
                                                   ])
        return train_data_transform
    else:
        test_data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize((224, 224)),
                                                  transforms.Normalize(t, p)
                                                  ])
        return test_data_transform


def batch_size_volume(volume, train_dataset, val_dataset):
    for batch in range(100):
        batch = batch + volume
        if min(len(train_dataset) % batch, len(val_dataset) % batch) > 50:
            break
        return batch
