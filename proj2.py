import os
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader

from pose_estimation import FrameData
from data_augs import *
from models.GaitPart.GaitPart import GaitPart
from loss import TripletLoss
from training import train
from evaluation import evaluate

# Nicholas Lower
# CS 722 HW4

'''
    Full project 2 file. Creates, trains, then evaluates GaitPart model.

    Uses pre-stored training and testing sets found in data directory.
'''

# Some basic augmentations to diversify training dataset
training_tforms = transforms.Compose([
    TrimSequence(60),
    ToArray(),
    NormSequence(),
    FlipSequence(0.5),
    MirrorPoses(0.5),
    ShuffleSequence(0.5),
    ToTensor()])

# Simple, predefined transformations for test data (uses nothing from training data)
test_tforms = transforms.Compose([
    TrimSequence(60),
    ToArray(),
    NormSequence(),
    ToTensor()])

# Load training data. Already extracted and stored, so use load constructor.
# Two Noise Transform duplicates all data but with different (random) augmentations, used for contrast comparisons
train_data = FrameData(load=True, path=os.path.join('.', 'data', 'full_train_nearest.pickle'),
                      transforms=training_tforms)

# Load test data. Already extracted and stored, so use load constructor.
test_data = FrameData(load=True, path=os.path.join('.', 'data', 'full_test_nearest.pickle'),
                     transforms=test_tforms)

test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

# Model is GaitPart, with all optimal modules/structures as defined in paper.
model = GaitPart()

loss = TripletLoss()

if torch.cuda.is_available():
    model.to('cuda')
    loss.to('cuda')

# Train model
print('----------------------\n')
print('     TRAINING         \n')
print('----------------------\n')
eps, train_losses, valid_losses = train(train_data, model, loss, epochs=50, wd=0.0, lr=0.001)

# Plot learning curve
plt.plot(eps, train_losses, 'b', label='Training Loss')
plt.plot(eps, valid_losses, 'r', label = 'Validation Loss')
plt.title("GaitPart Implementation Learning Curve")
plt.legend()
plt.show()

# Evaluate model
print('----------------------\n')
print('     EVALUATION       \n')
print('----------------------\n')
evaluate(test_loader, model)

