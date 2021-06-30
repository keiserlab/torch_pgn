import os
import os.path as osp
import pandas as pd
import numpy as np
import sys

import copy

from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.functional as F

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib

sys.path.insert(0, "/srv/home/zgaleday/pgn")

from pgn.train.train_utils import make_save_directories, save_checkpoint
from pgn.data.ProximityGraphDataset import ProximityGraphDataset
from pgn.data.FingerprintDataset import FingerprintDataset
from pgn.data.data_utils import parse_transforms
from pgn.models.pfp_encoder import PFPEncoder
from pgn.models.FPEncoder import FPEncoder
from scripts.pair_networks import load_args

from torch.utils.tensorboard import SummaryWriter

LABEL_FILE = '/srv/home/zgaleday/IG_data/raw_data/d4_test_compounds/experimally_test_chunkmap.csv'


def run_classifier(checkpoint_path, dataset_path, savedir, device, epochs, repeats=5):
    for iter in range(repeats):
        args, full_dataset_path = load_args(checkpoint_path, dataset_path, savedir, device, epochs)
        base_dir = savedir
        save_dir = osp.join(base_dir, 'repeat_{0}'.format(iter))
        os.mkdir(save_dir)
        args.save_dir = save_dir
        make_save_directories(save_dir)
        args.seed = np.random.randint(0, 1e4)
        torch.manual_seed(args.seed)
        if args.encoder_type == 'fp':
            train_dataset = ClassificationFPDataset(args, LABEL_FILE)
            train_idx, val_idx = train_dataset.generate_split_indices()
            val_dataset = train_dataset[val_idx]
            train_dataset = train_dataset[train_idx]
        else:
            train_dataset = ClassificationProximityGraphDataset(args, LABEL_FILE)
            train_idx, val_idx = train_dataset.generate_split_indices()
            val_dataset = train_dataset[val_idx]
            train_dataset = train_dataset[train_idx]

            dist_mean, dist_std = args.distance_mean, args.distance_std

            train_dataset.data.edge_attr[:, 0] = (train_dataset.data.edge_attr[:, 0] - dist_mean) / dist_std
            val_dataset.data.edge_attr[:, 0] = (val_dataset.data.edge_attr[:, 0] - dist_mean) / dist_std

            transforms = parse_transforms(args.transforms)
            train_dataset.data = transforms(train_dataset.data)

            args.node_dim, args.edge_dim = train_dataset.getNodeDim(), train_dataset.getEdgeDim()

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = ClassificationNetwork(args)
        model.to(args.device)
        criterion = nn.BCELoss()
        lr = 3e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

        model = train(model, optimizer, train_dataloader, val_dataloader, args, criterion)

        evaluate_classifier_experimental(model, args, train_idx, val_idx)

        calculate_classifier_confusion_matrix(args)


def calculate_classifier_confusion_matrix(args):
    experimental_path = osp.join(args.save_dir, 'experimental_df.csv')
    experimental_df = pd.read_csv(experimental_path)
    binary_predictions = experimental_df['predicted'].values
    binary_predictions[binary_predictions < 0.5] = 0
    binary_predictions[binary_predictions >= 0.5] = 1
    experimental_df['predicted'] = binary_predictions
    train = experimental_df[experimental_df['set'] == 'train']
    test = experimental_df[experimental_df['set'] == 'test']
    cm_test = confusion_matrix(test['labels'], test['predicted'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Non Binder", "Binder"])
    disp = disp.plot()
    plt.tight_layout()
    disp.ax_.tick_params(axis='both', which='major', labelsize='8')
    plt.savefig(osp.join(args.save_dir, 'results', 'classifier_CM_test.png'))
    plt.close()
    cm_train = confusion_matrix(train['labels'], train['predicted'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["Non Binder", "Binder"])
    disp = disp.plot()
    plt.tight_layout()
    disp.ax_.tick_params(axis='both', which='major', labelsize='8')
    plt.savefig(osp.join(args.save_dir, 'results', 'classifier_CM_train.png'))
    plt.close()


def evaluate_classifier_experimental(model, args, train_idx, val_idx):
    if args.encoder_type != 'fp':
        train_examples = ProximityGraphDataset(args)[list(train_idx)]
        test_examples = ProximityGraphDataset(args)[list(val_idx)]

        dist_mean, dist_std = args.distance_mean, args.distance_std

        train_examples.data.edge_attr[:, 0] = (train_examples.data.edge_attr[:, 0] - dist_mean) / dist_std
        test_examples.data.edge_attr[:, 0] = (test_examples.data.edge_attr[:, 0] - dist_mean) / dist_std

        transforms = parse_transforms(args.transforms)

        train_examples.data = transforms(train_examples.data)
        test_examples.data = transforms(test_examples.data)

    else:
        train_examples = FingerprintDataset(args)[list(train_idx)]
        test_examples = FingerprintDataset(args)[list(val_idx)]

    train_support_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_examples, batch_size=args.batch_size, shuffle=False)

    train_name, test_name = [], []
    train_predicted, test_predicted = [], []
    with torch.no_grad():
        model.eval()
        for batch in train_support_dataloader:
            batch.to(args.device)
            fp = model(batch).detach().cpu().numpy()
            name = batch.name
            train_predicted.append(fp)
            train_name.append(name)
        for batch in test_dataloader:
            batch.to(args.device)
            fp = model(batch).detach().cpu().numpy()
            name = batch.name
            test_predicted.append(fp)
            test_name.append(name)
    train_predicted = np.concatenate(train_predicted)
    test_predicted = np.concatenate(test_predicted)
    train_name = np.concatenate(train_name)
    test_name = np.concatenate(test_name)

    activity_labels = pd.read_csv(LABEL_FILE)[['ZINC ID', 'Binder or not']]
    train_activity_label = []
    for name in train_name:
        train_activity_label.append(
            np.array(activity_labels[activity_labels['ZINC ID'] == name]['Binder or not'].head())[0])
    test_activity_label = []
    for name in test_name:
        test_activity_label.append(
            np.array(activity_labels[activity_labels['ZINC ID'] == name]['Binder or not'].head())[0])

    train_labels = ['train'] * len(train_examples)
    test_labels = ['test'] * len(test_examples)

    names = np.concatenate([train_name, test_name])
    predicted = np.concatenate([train_predicted, test_predicted])
    labels = np.concatenate([train_activity_label, test_activity_label])
    group = np.concatenate([train_labels, test_labels])

    experimental_df = pd.DataFrame({'predicted': predicted[:, 0], 'labels': labels, 'set': group, 'name': names})
    experimental_df.to_csv(osp.join(args.save_dir, 'experimental_df.csv'), index=False)


def evaluate_classifier_full(model, args, full_dataset_path):

    args.data_path = full_dataset_path
    if args.encoder_type != 'fp':
        full_dataset = ProximityGraphDataset(args)

        dist_mean, dist_std = args.distance_mean, args.distance_std

        full_dataset.data.edge_attr[:, 0] = (full_dataset.data.edge_attr[:, 0] - dist_mean) / dist_std
        transforms = parse_transforms(args.transforms)

        full_dataset.data = transforms(full_dataset.data)
    else:
        full_dataset = FingerprintDataset(args)

    full_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    full_name = []
    full_predicted = []
    with torch.no_grad():
        model.eval()
        for batch in full_dataloader:
            batch.to(args.device)
            fp = model(batch).detach().cpu().numpy()
            name = batch.name
            full_predicted.append(fp)
            full_name.append(name)

    full_predicted = np.concatenate(full_predicted)
    full_name = np.concatenate(full_name)

    full_df = pd.DataFrame({'predicted': full_predicted[:, 0], 'name': full_name})
    full_df.to_csv(osp.join(args.save_dir, 'full_df.csv'), index=False)


def train(model, optimizer, train_loader, val_loader, args, criterion):
    train_losses = []
    val_losses = []
    step = 0
    best_loss = float('inf')
    best_params = model.state_dict()
    writer = SummaryWriter(log_dir=args.save_dir)
    for epoch in tqdm(range(args.epochs)):
        loss = 0.
        model.train()
        for data in train_loader:
            data.to(args.device)
            optimizer.zero_grad()
            labels = data.y.to(args.device)
            output = model(data)
            local_loss = criterion(output.squeeze(), labels)
            local_loss.backward()
            optimizer.step()
            loss += local_loss.item()
        avg_train_loss = loss / len(train_loader)
        writer.add_scalar("Training loss", avg_train_loss, epoch + 1)
        val_loss = 0.
        with torch.no_grad():
            model.eval()
            for data in val_loader:
                data.to(args.device)
                labels = data.y.to(args.device)
                output = model(data)
                local_loss = criterion(output.squeeze(), labels)
                val_loss += local_loss.item()
        if val_loss < best_loss:
            best_params = model.state_dict()
            save_checkpoint(osp.join(args.save_dir, 'best_checkpoint.pt'), model, args)
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Validation loss", avg_val_loss, epoch+1)
        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
            .format(epoch+1, args.epochs, avg_train_loss, avg_val_loss))
    model.load_state_dict(best_params)
    return model


class ClassificationFPDataset(FingerprintDataset):
    def __init__(self, args, binding_file, val_percent=0.2, transform=None, pre_transform=None):
        self.binding_file = binding_file
        self.args = args
        self.val_percent = val_percent
        super(ClassificationFPDataset, self).__init__(args, transform=transform, pre_transform=pre_transform)
        self.load_binding()

    def load_binding(self):
        activity_labels = pd.read_csv(self.binding_file)[['ZINC ID', 'Binder or not']]
        binder_index = []
        non_binder_index = []
        for i, name in enumerate(self.data.name):
            binder_label = int(activity_labels[activity_labels['ZINC ID'] == name]['Binder or not'].head(1))
            if binder_label == 0:
                non_binder_index.append(i)
            else:
                binder_index.append(i)
            self.data.y[i] = binder_label
        # self.data.y = self.data.y.type(torch.LongTensor)
        self.categories = {0: non_binder_index, 1: binder_index}

    def generate_split_indices(self):
        num_binder = len(self.categories[1])
        num_non_binder = len(self.categories[0])
        rand = np.random.RandomState(self.args.seed)
        binder_permutations = rand.permutation(num_binder)
        non_binder_permutations = rand.permutation(num_non_binder)
        val_binders = np.array(self.categories[1])[binder_permutations[:int(num_binder * self.val_percent)]]
        train_binders = np.array(self.categories[1])[binder_permutations[int(num_binder * self.val_percent):]]
        val_non_binders = np.array(self.categories[0])[non_binder_permutations[:int(num_non_binder * self.val_percent)]]
        train_non_binders = np.array(self.categories[0])[
            non_binder_permutations[int(num_non_binder * self.val_percent):]]
        val_indices = np.concatenate([val_binders, val_non_binders])
        train_indices = np.concatenate([train_binders, train_non_binders])
        return list(train_indices), list(val_indices)


class ClassificationProximityGraphDataset(ProximityGraphDataset):
    def __init__(self, args, binding_file, val_percent=0.2, transform=None, pre_transform=None):
        self.binding_file = binding_file
        self.args = args
        self.val_percent = val_percent
        super(ClassificationProximityGraphDataset, self).__init__(args, transform=transform,
                                                                  pre_transform=pre_transform)
        self.load_binding()

    def load_binding(self):
        activity_labels = pd.read_csv(self.binding_file)[['ZINC ID', 'Binder or not']]
        binder_index = []
        non_binder_index = []
        for i, name in enumerate(self.data.name):
            binder_label = int(activity_labels[activity_labels['ZINC ID'] == name]['Binder or not'].head(1))
            if binder_label == 0:
                non_binder_index.append(i)
            else:
                binder_index.append(i)
            self.data.y[i] = binder_label
        # self.data.y = self.data.y.type(torch.LongTensor)
        self.categories = {0: non_binder_index, 1: binder_index}

    def generate_split_indices(self):
        num_binder = len(self.categories[1])
        num_non_binder = len(self.categories[0])
        rand = np.random.RandomState(self.args.seed)
        binder_permutations = rand.permutation(num_binder)
        non_binder_permutations = rand.permutation(num_non_binder)
        val_binders = np.array(self.categories[1])[binder_permutations[:int(num_binder * self.val_percent)]]
        train_binders = np.array(self.categories[1])[binder_permutations[int(num_binder * self.val_percent):]]
        val_non_binders = np.array(self.categories[0])[non_binder_permutations[:int(num_non_binder * self.val_percent)]]
        train_non_binders = np.array(self.categories[0])[
            non_binder_permutations[int(num_non_binder * self.val_percent):]]
        val_indices = np.concatenate([val_binders, val_non_binders])
        train_indices = np.concatenate([train_binders, train_non_binders])
        return list(train_indices), list(val_indices)

    def getNodeDim(self):
        return self.data.x.numpy().shape[1]

    def getEdgeDim(self):
        return self.data.edge_attr.numpy().shape[1]


class ClassificationNetwork(nn.Module):
    def __init__(self, args):
        super(ClassificationNetwork, self).__init__()
        self.args = args
        if self.args.encoder_type == 'fp':
            self.encoder = FPEncoder(args)
        else:
            self.encoder = PFPEncoder(args, args.node_dim, args.edge_dim)
        self.fc_1 = nn.Linear(args.fp_dim, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, d1):
        fp1 = self.encoder.forward(d1)
        fp1 = self.dropout(self.activation(self.fc_1(fp1)))
        fp1 = self.fc_2(fp1)
        return self.sigmoid(fp1)


