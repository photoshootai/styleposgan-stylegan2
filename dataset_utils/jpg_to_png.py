import os
import pickle

jpg_pkl_path = "./data/CombinedDataset2Training.pkl"
png_pkl_path = "./data/CombinedDataset2TrainingPng.pkl"

with open(jpg_pkl_path, 'rb') as f:
    jpg_pairs = pickle.load(f)


new_pairs =[]
for (a, b) in jpg_pairs:
    a = a.replace(".jpg", ".png")
    b = b.replace(".jpg", ".png")
    new_pairs.append((a, b))

with open(png_pkl_path, 'wb') as f:
    pickle.dump(new_pairs, f)

