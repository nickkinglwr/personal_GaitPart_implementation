import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Nicholas Lower
# Training stage in GaitPart model pipeline.
# Original paper - doi: 10.1109/CVPR42600.2020.01423

def train(data, model, criterion, epochs=1, lr=0.0001, wd=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_losses = []
    valid_losses = []

    for e in range(epochs):
        print('========= Epoch {} ========='.format(e))
        model.train()
        avg_loss = 0

        valid, training = torch.utils.data.random_split(data, [len(data)//5, len(data) - len(data)//5]) # Split into training and validation set
        train_loader = DataLoader(training, shuffle=True, batch_size=32)
        valid_loader = DataLoader(valid, shuffle=True, batch_size=32)

        length = len(train_loader)

        for frames, subject in train_loader:
            frames.unsqueeze_(2)

            # Send data to cuda device if available
            if torch.cuda.is_available():
                frames = frames.to('cuda', non_blocking=True)
                subject = subject.to('cuda', non_blocking=True)

            # Get final feature embeddings from model for both inputs
            final_features = model(frames)

            # Calculate loss
            # Loss for GaitPart is 16 separate values, one loss for each part
            loss = criterion(final_features, subject[:])

            # Update model
            # Since loss is 16 separate losses for each part, use mean for total loss
            loss.mean().backward()
            avg_loss += loss.mean()
            opt.step()
            opt.zero_grad()

        avg_loss /= length
        print("***** Average training loss: {} *****".format(avg_loss))
        train_losses.append(avg_loss.item())

        # Use validation dataset to get validation loss
        valid_losses.append(validation(valid_loader, model, criterion))

    return range(epochs), train_losses, valid_losses


def validation(validation_data, model, criterion):
    model.eval()
    avg_loss = 0
    length = len(validation_data)

    with torch.no_grad():
        for frames, subject in validation_data:
            frames.unsqueeze_(2)

            if torch.cuda.is_available():
                frames = frames.to('cuda', non_blocking=True)
                subject = subject.to('cuda', non_blocking=True)

            final_features = model(frames)
            loss = criterion(final_features, subject[:])
            avg_loss += loss.mean()

    avg_loss /= length
    print("***** Average validation loss: {} *****".format(avg_loss))
    return avg_loss.item()


def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    import pose_estimation
    from loss import TripletLoss
    from models.GaitPart import GaitPart
    from data_augs import *

    transforms = transforms.Compose([
        TrimSequence(60),
        ToArray(),
        NormSequence(),
        FlipSequence(0.5),
        MirrorPoses(0.5),
        ShuffleSequence(0.5),
        ToTensor()])

    # Load data. Already extracted and stored, so use load constructor.
    # Two Noise Transform duplicates all data but with different (random) augmentations, used for contrast comparisons
    train_data = pose_estimation.FrameData(load=True, path=os.path.join('.', 'data','sil_train_nearest.pickle'),
                                           transforms=TwoNoiseTransform(transforms))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=16)

    model = GaitPart.GaitPart()

    criterion = TripletLoss()

    if torch.cuda.is_available():
        model.to('cuda')
        criterion.to('cuda')

    train(train_loader, model, criterion, epochs=100)

    save_model(model, os.path.join('.', 'saved models','saved_model_trained.pth'))
