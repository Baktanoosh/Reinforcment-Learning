from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        self.init_tensor = torchvision.transforms.ToTensor()
        with open(dataset_path+"/"+"labels.csv") as labels_csv_file:
            label_reader = csv.reader(labels_csv_file)
            next(label_reader) 
            self.input_data = []
            for row in label_reader:
                file_name = row[0]
                label = row[1]
                j = 0
                for i in LABEL_NAMES:
                  if  label == i:
                    label_index = j
                    break
                  j += 1
                image_tensor = self.init_tensor(Image.open(dataset_path+"/"+file_name))
                self.input_data.append((image_tensor,label_index))
    def __len__(self):
        self.length = len(self.input_data)
        return self.length


    def __getitem__(self, idx):
        self.item = self.input_data[idx]
        return self.item


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
