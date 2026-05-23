@echo off
sed -i 's/from torch.utils.data import Dataset, DataLoader/from torch.utils.data import Dataset/g' data_provider/data_loader.py
