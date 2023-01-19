import os
import numpy as np
import pickle

from torch.utils.data import Dataset
import cv2
import glob

# Nicholas Lower
# CS 722 HW4

class FrameData(Dataset):
    '''
    Extract and store frame data from CASIA-B Silhouette Dataset.

    Args:
        subs - Tuple of CASIA-B subjects to extract, predict, and store.
        types - Tuple of CASIA-B types to extract. Must be subset of ('bg','cl','nm').
        seq - Tuple of CASIA-B sequence numbers to extract for each subject.
        angles - Tuple of CASIA-B angles to extract. Must be subset of (0,18,36,54,...,180).
        load - Boolean to signal that the dataset should be loaded from previous pickling stored at path.
        path - File path for loading previous stored dataset. Useless if load = False.
        transforms - Torchvision augmentation transforms, applied when fetching data (__getitem__)
        train - Signals preparation of training data or not, used in saving label data.

    Stores frame data in self.data and subject labels in self.labels, same index indicates pair.
    '''
    def __init__(self, subs = tuple(range(1,125)), types = ('bg','cl','nm'), seq = tuple(range(1,7)), angles = tuple(range(0,198,18)),
                 load = False, path = None, transforms = None, train = True):

        if not load:
            self.subs = subs
            self.types = types
            self.seq = seq
            self.angles = angles
            self.data = []
            self.labels = []
            self.transforms = transforms
            self.train = train

            self.get_frames(subs, types, seq, angles)

        else:
            self.load_dataset(path)
            # Store new transformations if any
            if transforms is not None:
                self.transforms = transforms

    def get_frames(self, subs, types, seqs, angs):
        path = ['./data/GaitDatasetB-silh','','','']
        for sub in subs:
            path[1] = str(sub).zfill(3)
            for type in types:
                for seq in seqs:
                    path[2] = '-'.join([type, str(seq).zfill(2)])
                    if (type == 'bg' or type == 'cl') and seq > 2:
                        break
                    for ang in angs:
                        path[3] = str(ang).zfill(3)
                        full_path = '/'.join(path) + '/*.png'
                        frames = []
                        for f in glob.glob(full_path):
                            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                            contours,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                            mx = (0, 0, 0, 0)  # biggest bounding box so far
                            mx_area = 0
                            for cont in contours:
                                x, y, w, h = cv2.boundingRect(cont)
                                area = w * h
                                if area > mx_area:
                                    mx = x, y, w, h
                                    mx_area = area
                            x, y, w, h = mx

                            crop = img[y:y+h,x:x+w]
                            if crop.size > 0:
                                res = cv2.resize(crop, dsize=(44, 64), interpolation=cv2.INTER_NEAREST)
                                frames.append(np.array(res))

                        if len(frames) > 0:
                            self.data.append(frames)
                            if self.train:
                                self.labels.append(sub)
                            else:
                                self.labels.append((sub, type, seq, ang))

    def save_dataset(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_dataset(self, path):
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.labels[item]

        if self.transforms is not None:
            data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # Get full training data (subjects 1 - 74) and save to data directory
    train_pd = FrameData(subs=tuple(range(1,75)))
    train_pd.save_dataset(os.path.join('.', 'data','full_train_nearest.pickle'))
    train_pd = None

    # Get full testing data (subjects 75 - 124) and save to data directory
    test_pd = FrameData( subs=tuple(range(75,125)), train = False)
    test_pd.save_dataset(os.path.join('.', 'data','full_test_nearest.pickle'))