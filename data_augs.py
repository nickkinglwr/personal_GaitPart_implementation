import numpy as np
import torch

# Nicholas Lower
# CS 722 HW4

class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data.copy(), dtype=torch.float)

class ToArray(object):
    def __call__(self, data):
        if len(data) == 0:
            return
        if isinstance(data, list):
            return np.stack(data, 0)
        return data

class TrimSequence(object):
    '''
        Make normal 100 frame sequence smaller by removing frames and start and end, if sequence is smaller than required
        number then duplicate sequence then trim.

        Makes classification easier because pose estimation of far away subjects at very start and very end is volatile,
        also makes computation cheaper.
    '''
    def __init__(self, frames_to_trim_to = 60):
        self.frames = frames_to_trim_to

    def __call__(self, seq):
        l = len(seq)
        if l > self.frames:
            frames_trim = l - self.frames
            if frames_trim % 2 == 0:
                return seq[frames_trim//2: l - (frames_trim//2)]
            else:
                return seq[frames_trim // 2: l - (frames_trim // 2 + 1)]


        while l < self.frames:
            seq.extend(seq[:self.frames - l])
            l = len(seq)

        return seq


class NormSequence(object):
    def __call__(self, seq):
        return seq / 255


class MatchSizes(object):
    def __call__(self, seq):
        return [np.resize(a,(64,44)) for a in seq]


class MirrorPoses(object):
    '''
        Mirror poses across joint's center of gravity, has effect of walking opposite direction
    '''
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:

            data = np.flip(data, 2)

        return data

class FlipSequence(object):
    '''
        Flip sequence of poses, has effect of walking backwards
    '''
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data, axis=0)
        return data

class ShuffleSequence(object):
    '''
        Shuffle frames in a gait sequence.
    '''
    def __init__(self, enabled=False):
        self.enabled = enabled

    def __call__(self, data):
        if self.enabled:
            np.random.shuffle(data)
        return data


class TwoNoiseTransform(object):
    """
        Create two versions of the same pose
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class DropOutFrames(object):
    """
    Drop frames randomly from a sequence.
    """

    def __init__(self, probability=0.1, sequence_length=60):
        self.probability = probability
        self.sequence_length = sequence_length

    def __call__(self, data):
        T, V, C = data.shape

        new_data = []
        dropped = 0
        for i in range(T):
            if np.random.random() <= self.probability:
                new_data.append(data[i])
            else:
                dropped += 1
            if T - dropped <= self.sequence_length:
                break

        for j in range(i, T):
            new_data.append(data[j])

        return np.array(new_data)


class DropOutJoints(object):
    """
        Zero joints randomly from a pose.
    """

    def __init__(
        self, prob=1, dropout_rate_range=0.1,
    ):
        self.dropout_rate_range = dropout_rate_range
        self.prob = prob

    def __call__(self, data):
        if np.random.binomial(1, self.prob, 1) != 1:
            return data

        T, V, C = data.shape
        data = data.reshape(T * V, C)
        # Choose the dropout_rate randomly for every sample from 0 - dropout range
        dropout_rate = np.random.uniform(0, self.dropout_rate_range, 1)
        zero_indices = 1 - np.random.binomial(1, dropout_rate, T * V)
        for i in range(3):
            data[:, i] = zero_indices * data[:, i]
        data = data.reshape(T, V, C)
        return data


class MultiInput(object):
    '''
        Construct multiple inputs from joint data.

        Specifically, construct joints, bones, and velocity data.
    '''
    def __init__(self, connect_joint, enabled=False):
        self.connect_joint = connect_joint
        self.enabled = enabled

    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))

        if not self.enabled:
            return data[np.newaxis, ...]

        C, T, V = data.shape
        data_new = np.zeros((3, C * 2, T, V))

        # Joints
        data_new[0, :C, :, :] = data
        for i in range(V):
            data_new[0, C:, :, i] = data[:, :, i] - data[:, :, 1]

        # Velocity
        for i in range(T - 2):
            data_new[1, :C, i, :] = data[:, i + 1, :] - data[:, i, :]
            data_new[1, C:, i, :] = data[:, i + 2, :] - data[:, i, :]

        # Bones
        for i in range(len(self.connect_joint)):
            data_new[2, :C, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        bone_length = 0
        for i in range(C - 1):
            bone_length += np.power(data_new[2, i, :, :], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            data_new[2, C, :, :] = np.arccos(data_new[2, i, :, :] / bone_length)

        return data_new