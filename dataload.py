import pandas as pd
import os
from tqdm.notebook import tqdm
path = '/content/drive/MyDrive/ManasLabDay2'
header = ['timestamp', 'voc2', 'no2', 'eth', 'co', 'temp', 'pressure', 'humidity', 'mq3', 'mq7', 'mq9', 'mq135']

def checkHeader(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        # Read the first row of the CSV file
        first_row = csvfile.readline().strip()

        # Check if the first row matches the header
        if first_row != ','.join(header):
            # Read the rest of the file
            csvfile.seek(0)
            rows = csvfile.readlines()

            # Prepend the header to the rows
            rows.insert(0, ','.join(header) + '\n')

            # Write the updated rows back to the file
            with open(file_path, 'w', newline='') as csvfile:
                csvfile.writelines(rows)

fold_path = path
save_path = '/content/drive/MyDrive/combined_data/'
os.makedirs(save_path, exist_ok=True)


data_path = '/content/drive/MyDrive/combined_data/'

# st = set()
ls = os.listdir(data_path)
ls = [f.replace('.csv', '') for f in ls]
ls = [''.join(char for char in f if not char.isdigit()) for f in ls ]
ls = {key : value for value, key in enumerate(sorted(list(set(ls))))}

print(ls)
len(ls)



class SmellDataset(Dataset):
  def __init__(self, data_path, t_lim=3000):

    # self.classToLabel = label
    self.data_path = data_path
    self.files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    # pprint(self.files)
    # checked works
    self.len = len(self.files)
    self.t_lim = t_lim
    ls = os.listdir(data_path)
    ls = [f.replace('.csv', '') for f in ls]
    ls = [''.join(char for char in f if not char.isdigit()) for f in ls ]
    ls = {key : value for value, key in enumerate(sorted(list(set(ls))))}


    self.classToLabel = ls

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    file_path = os.path.join(self.data_path, self.files[idx])
    df = pd.read_csv(file_path, nrows = self.t_lim)
    # df.drop_columns(['timestamp', ''])
    df.drop(columns=['timestamp', 'Unnamed: 0'],inplace = True)
    # pprint(df.head())

    fname = self.files[idx].split('.')[0]
    # key = [char for char in fname if char is not char.isdigit() else ]
    key = [char for char in fname if not char.isdigit() ]
    # print(key)
    key = "".join(key)
    # print(self.files[idx].split('.')[0])

    y = self.classToLabel[key]
    return df.to_numpy(dtype =np.float32), y



dataset = SmellDataset(data_path)
print(len(dataset))
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)


X, y = next(iter(loader))
print(X.shape, y.dtype)

# ((N, T, C) , N)