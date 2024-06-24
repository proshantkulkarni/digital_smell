import os
import pandas as pd
import numpy as np



# print(path_dict['Polo'])

def sample_from_dataframe(df):
    window_len = 100
    sample_arr = np.array(df)
    sample_ls = []
    for i in range(600,3000 - window_len):
        sample_ls.append(sample_arr[i:i+window_len])
    samples = np.stack(sample_ls)
    return samples


def combine_files(path_dict):
    for key in path_dict.keys():
        path_list = path_dict[key]
        samples = []
        for local_path in path_list:
            df = pd.DataFrame()
            files = os.listdir(local_path)
            for file in files:
                file_path = os.path.join(local_path, file)
                local_df = pd.read_csv(file_path)
                df = pd.concat([df, local_df], axis = 0)
                df = df.drop(columns = 'timestamp')
            local_samples = sample_from_dataframe(df)
            samples.append(local_samples)
            # print(len(samples))
            # print(type(samples[0]))
        if len(samples)>1:
            samples = np.concatenate(samples, axis=0)
        else:
            samples = np.array(samples)
        sample_label = np.full((samples.shape[0],),key)
        print(f'{key} data shape: {samples.shape}, label shape: {sample_label.shape}')
        np.savez(f'curated_data/data_{key}.npz', data1=samples, data2=sample_label)
        # cls_df = pd.DataFrame(dir_samples, columns=['voc2','no2','eth','co','temp','pressure','humidity','mq3','mq7','mq9','mq135'])
        # cls_df.loc[:,'label'] = sample_label
        # cls_df.to_csv(f'Combined_sampled_{key}.csv')


if __name__ == '__main__':
    class_list = ['Background0','Cardamom','Clinic_plus', 'Garlic', 'Ginger', 'Incense_stick', 'Onion', 'Polo']
    data_dir = 'ManasLabDay2'
    path_dict = dict()
    data_base_dir = os.path.join(os.getcwd(),data_dir)

    for root, dirs, files in os.walk(data_base_dir, topdown=False):
        for name in dirs:
            for cls_name in class_list:
                if (cls_name in name) or (cls_name is name):
                    if cls_name not in path_dict.keys():
                        path_dict[cls_name] = list()
                    else:
                        local_path = os.path.join(data_base_dir, name)
                        path_dict[cls_name].append(local_path)

    combine_files(path_dict)

    data_curated_dir = 'curated_data'

    # path_to_curated_data = os.path.join(os.getcwd(), data_curated_dir)
    # files = os.listdir(path_to_curated_data)
    # np