import os
import cv2
import torch
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class LSA64Dataset(Dataset):
    def __init__(self, video_dir, num_frames=12, img_size=224, transform=None):
        self.video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform

        self.labels = [os.path.basename(v).split('_')[0] for v in self.video_paths]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.encoded_labels[idx]
        frames = self.load_frames(path)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        return frames, label

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []

        for i in range(total):
            ret, frame = cap.read()
            if i in frame_idxs and ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return [torch.zeros(3, self.img_size, self.img_size)] * self.num_frames

        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))

        frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
        return frames
