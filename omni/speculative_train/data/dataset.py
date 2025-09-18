# edit from specforge/data/preprocessing.py

import os
import torch

def list_local_files(path, suffix):
    datapaths = []
    for root, directories, files in os.walk(path):
        part_datapaths = [os.path.join(root, file) for file in files if file.endswith(suffix)]
        datapaths.extend(part_datapaths)
    return datapaths

class OfflineEagleDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None, max_len=2048):
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len

    def __len__(self):
        return len(self.datapaths)

    def _open_file(self, index):
        return torch.load(self.datapaths[index], weights_only=False)

    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            print(f"ERROR Failed to load {self.datapaths[index]} with error {e}")
            data = self._open_file(0)
            # raise e
        new_data = {}

        # Squeeze due to our data generation script adding a batch dimension
        hidden_states = data["hidden_states"][:self.max_len]

        input_ids = data["input_ids"][1: self.max_len + 1]
        loss_mask = data["loss_mask"][: self.max_len]
        loss_mask[-1] = 0

        pad_size = self.max_len - input_ids.numel()
        pad_zeros = torch.zeros(pad_size, dtype=input_ids.dtype, device=input_ids.device)

        new_data["loss_mask"] = torch.cat([loss_mask, pad_zeros.to(loss_mask.dtype)]).view(1, -1)
        new_data["attention_mask"] = torch.ones_like(new_data["loss_mask"], dtype=torch.long)
        new_data["hidden_states"] = torch.cat(
            [hidden_states, torch.zeros(pad_size, hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)],
            dim=0,
        ).unsqueeze(0)
        new_data["input_ids"] = torch.cat([input_ids, pad_zeros]).view(1, -1)
        if self.transform:
            new_data = self.transform(new_data)

        return new_data

    def set_epoch(self, epoch):
        self._epoch = epoch

def build_offline_eagle_dataset(
    hidden_states_path: str,
    max_len: int,
    suffix: str,
) -> torch.utils.data.Dataset:
    return OfflineEagleDataset(
        list_local_files(hidden_states_path, suffix),
        max_len=max_len,
    )
