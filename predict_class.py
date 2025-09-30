import os
import pandas as pd
import torch
from Model_Swin_transformer import swin_tiny_patch4_window7_224 as create_model
import warnings

warnings.filterwarnings("ignore")
from utils_class import Dataset_Loader, make_test_data, data_transformer, binary_classifier_predict


def predict_software(Dir, Clinical_data, train_val_test_cohort_list, model_path, save_file_name):
    test_image_path, test_patient_name, test_patient_label, test_patient_follow_up_time = \
        make_test_data(data_path=Dir,
                       test_cohort_list=train_val_test_cohort_list,
                       Clinical_data=Clinical_data)

    test_dataset = Dataset_Loader(image_path=test_image_path,
                                  patient_name=test_patient_name,
                                  patient_label=test_patient_label,
                                  patient_follow_up_time=test_patient_follow_up_time,
                                  transform=data_transformer(split='test'))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=16,
                                              collate_fn=test_dataset.collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    sum_loss, sum_auc, predict_data = binary_classifier_predict(model=model,
                                                                data_loader=test_loader,
                                                                device=device,
                                                                epoch=0,
                                                                part='test')
    predict_data.to_csv(os.path.join(Dir, save_file_name), encoding="gbk", index=False)


def main():
    Dir = r''
    save_file_name = '.csv'
    Clinical_data_path = os.path.join(Dir, "Clinical_ifo.csv")
    Clinical_data = pd.read_csv(Clinical_data_path, index_col=0, encoding='gbk')
    Clinical_data = Clinical_data.iloc[:, :3]
    Clinical_data.columns = ["p_ID", "fustat", "futime"]
    train_val_test_cohort_list = ['']
    predict_weights_path = r'image_path\weight\.pth'

    predict_software(Dir=Dir,
                     Clinical_data=Clinical_data,
                     train_val_test_cohort_list=train_val_test_cohort_list,
                     model_path=predict_weights_path,
                     save_file_name=save_file_name)
    print('finish !!!')


if __name__ == '__main__':
    main()
