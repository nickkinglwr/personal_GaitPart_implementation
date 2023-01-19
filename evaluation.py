import torch
import numpy as np

from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader


# Nicholas Lower
# CS 722 HW4

def evaluate(data, model):
    model.eval()
    full_outs = {}

    with torch.no_grad():
        for frames, label in data:
            frames.unsqueeze_(2)
            if torch.cuda.is_available():
                frames = frames.cuda()

            # Compute final features of current testing sample
            out = model(frames)

            # For every subject, type, sequence, and angle in batch store in final outputs table
            # Extended list comprehension needed because some parts of label are Tensors other tuples
            for i in range(frames.shape[0]):
                full_outs[tuple(l[i].item() if type(l) is torch.Tensor else l[i] for l in label)] = out[i].cpu().numpy()

    # Split all outputs into gallery and probe sets.
    # First 4 sequences (for all 11 angles) of each subject are gallery, rest are probe
    gallery_outs = []
    gallery_labels = []
    probe = []
    for k, v in full_outs.items():
        if k[2] >= 4:
            gallery_outs.append(v)
            gallery_labels.append(k)
        else:
            probe.append((k, v))

    gallery_outs = np.array(gallery_outs)
    correct = 0
    total = 0

    for target, output in probe:
        # Find gallery entry with shortest distance from current probe output embedding
        distance = np.linalg.norm(gallery_outs - output,axis=(1,2))
        min_target = gallery_labels[np.argmin(distance)]

        # If closest gallery subject is same as probe, then the model is correct
        if min_target[0] == target[0]:
            correct += 1
        total += 1

    accuracy = correct / total
    print("Rank 1 Accuracy: {:0.2f}%".format(accuracy * 100))



if __name__ == '__main__':
    import os
    from models.GaitPart import GaitPart
    from data_augs import *
    from pose_estimation import FrameData

    transforms = Compose([TrimSequence(60), ToArray(), NormSequence(), ToTensor()])

    test_data = FrameData(load=True, path=os.path.join('.', 'data', 'sil_test_nearest.pickle'),
                                          transforms=transforms)

    test_loader = DataLoader(test_data, batch_size=12,)

    model = GaitPart.GaitPart()

    if torch.cuda.is_available():
        model.cuda()

    evaluate(test_loader, model)

