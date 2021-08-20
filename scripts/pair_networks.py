import os
import os.path as osp
import pandas as pd
import numpy as np
import sys

import copy

from torch_geometric.data import DataLoader
import torch
import torch.nn as nn

from tqdm import tqdm
import re

from sklearn.metrics import plot_confusion_matrix
import joblib
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib

sys.path.insert(0, "/srv/home/zgaleday/pgn")

from pgn.train.train_utils import format_batch
from pgn.train.train_utils import load_checkpoint, make_save_directories, save_checkpoint
from pgn.data.ProximityGraphDataset import ProximityGraphDataset
from pgn.data.FingerprintDataset import FingerprintDataset
from pgn.data.data_utils import parse_transforms
from pgn.models.pfp_encoder import PFPEncoder
from pgn.models.FPEncoder import FPEncoder
from pgn.models.dmpnn_encoder import MPNEncoder
from torch.utils.tensorboard import SummaryWriter

LABEL_FILE = '/srv/home/zgaleday/IG_data/raw_data/d4_test_compounds/experimally_test_chunkmap.csv'


def run_pair_network(checkpoint_path, dataset_path, savedir, device, epochs, repeats=5, pre_trained=False):
    full_dataset_path = None
    for iter in range(repeats):
        args, full_dataset_path, stat_dict = load_args(checkpoint_path, dataset_path, savedir, device, epochs)
        base_dir = savedir
        save_dir = osp.join(base_dir, 'repeat_{0}'.format(iter))
        os.mkdir(save_dir)
        args.save_dir = save_dir
        make_save_directories(save_dir)
        args.seed = np.random.randint(0, 1e4)
        torch.manual_seed(args.seed)
        if args.encoder_type == 'fp':
            train_dataset = PairFPDataset(args, LABEL_FILE)
            val_dataset = PairFPDataset(args, LABEL_FILE)
        else:
            train_dataset = PairProximityGraphDataset(args, LABEL_FILE)
            val_dataset = PairProximityGraphDataset(args, LABEL_FILE)
            dist_mean, dist_std = args.distance_mean, args.distance_std

            train_dataset.data.edge_attr[:, 0] = (train_dataset.data.edge_attr[:, 0] - dist_mean) / dist_std
            val_dataset.data.edge_attr[:, 0] = (val_dataset.data.edge_attr[:, 0] - dist_mean) / dist_std

            transforms = parse_transforms(args.transforms)
            train_dataset.data = transforms(train_dataset.data)
            val_dataset.data = transforms(val_dataset.data)

        val_dataset.mode = 'validation'
        val_dataset.set_size = 100 * 16

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = SiameseNetwork(args)
        if pre_trained:
            model = set_encoder_weights(model, stat_dict)

        model.to(args.device)
        criterion = ContrastiveLoss()
        lr = 3e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

        model = train(model, optimizer, train_dataloader, val_dataloader, args, criterion)

        val_binders = train_dataset.val_binders
        train_binders = train_dataset.train_binders
        val_non_binders = train_dataset.val_non_binders
        train_non_binders = train_dataset.train_non_binders

        train_idx = np.concatenate([train_binders, train_non_binders])
        val_idx = np.concatenate([val_binders, val_non_binders])

        evaluate_pair_network_experimental(model, args, train_idx, val_idx)
        evaluate_pair_network_full(model, args, full_dataset_path)

        test_predict, test_labels, full_predict, classifier = classify_SVC(args)

        calculate_confusion_matrix(args.save_dir, classifier)
        save_classifications(args.save_dir, classifier)

    compute_metrics(savedir, checkpoint_path, dataset_path, device)


def set_encoder_weights(model, loaded_state_dict):
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r'(encoder\.encoder\.)([Wc])', loaded_param_name):
            param_name = loaded_param_name.replace('encoder.encoder', 'encoder.encoder.0')
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            print(f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            print(f'Warning: Pretrained parameter "{loaded_param_name}" '
                  f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            print(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    return model


def save_classifications(save_dir, classifier):
    experimental_path = osp.join(save_dir, 'experimental_df.csv')
    full_path = osp.join(save_dir, 'full_df.csv')
    experimental_df = pd.read_csv(experimental_path)
    experimental_df['predicted class'] = classifier.predict(experimental_df[['x', 'y']])
    experimental_df['probability binder'] = classifier.predict_proba(experimental_df[['x', 'y']])[:, 1]
    experimental_df.to_csv(osp.join(save_dir, 'results', 'experimental_classifications.csv'), index=False)
    full_df = pd.read_csv(full_path)
    full_df['predicted class'] = classifier.predict(full_df[['x', 'y']])
    full_df['predicted prob'] = classifier.predict_proba(full_df[['x', 'y']])[:,1]
    full_df.to_csv(osp.join(save_dir, 'results', 'full_classifications.csv'), index=False)


def compute_metrics(save_dir, checkpoint_path, experimental_ds_path, device):
    args, full_dataset_path, _ = load_args(checkpoint_path, experimental_ds_path, save_dir, device, 500)
    base_dir = args.save_dir
    count = 0
    AUCROCs = []
    APs = []
    CMs = []
    jaccards = []
    F1s = []
    Repeats = []
    for d in os.listdir(base_dir):
        if osp.isdir(osp.join(base_dir, d)):
            model_dir = osp.join(base_dir, d)
            args.save_dir = model_dir
            Repeats.append(d.split("_")[-1])
            classifier = joblib.load(osp.join(args.save_dir, 'model', 'svc_classifier.pkl'))
            count += 1
            experimental_classifications = pd.read_csv(
                osp.join(args.save_dir, 'results', 'experimental_classifications.csv'))
            test_set = experimental_classifications[experimental_classifications['set'] == 'test']
            # Calculate AUCROC
            AUCROCs.append(
                metrics.roc_auc_score(test_set['labels'], classifier.predict_proba(test_set[['x', 'y']])[:, 1]))
            disp = metrics.plot_roc_curve(classifier, test_set[['x', 'y']], test_set['labels'])
            plt.tight_layout()
            disp.ax_.tick_params(axis='both', which='major', labelsize='8')
            plt.savefig(osp.join(model_dir, 'results', 'classifier_AUCROC_test.png'))
            plt.close()
            # Calculate PRAUC
            APs.append(metrics.average_precision_score(test_set['labels'],
                                                       classifier.predict_proba(test_set[['x', 'y']])[:, 1]))
            disp = metrics.plot_precision_recall_curve(classifier, test_set[['x', 'y']], test_set['labels'])
            plt.tight_layout()
            disp.ax_.tick_params(axis='both', which='major', labelsize='8')
            plt.savefig(osp.join(model_dir, 'results', 'classifier_PRAUC_test.png'))
            plt.close()
            # Calculate confusion matrix
            CMs.append(metrics.confusion_matrix(test_set['labels'], classifier.predict(test_set[['x', 'y']])))
            disp = metrics.plot_confusion_matrix(classifier, test_set[['x', 'y']], test_set[['labels']],
                                                 display_labels=["Non Binder", "Binder"])
            plt.tight_layout()
            disp.ax_.tick_params(axis='both', which='major', labelsize='8')
            plt.savefig(osp.join(model_dir, 'results', 'classifier_CM_test.png'))
            plt.close()
            jaccards.append(metrics.jaccard_score(test_set['labels'], classifier.predict(test_set[['x', 'y']])))
            F1s.append(metrics.f1_score(test_set['labels'], classifier.predict(test_set[['x', 'y']])))
    CMs = np.array(CMs)
    AUCROCs = np.array(AUCROCs)
    APs = np.array(APs)
    jaccards = np.array(jaccards)
    F1s = np.array(F1s)
    metric_df = pd.DataFrame(
        {"Repeat": Repeats, "AUC ROC score": AUCROCs, "Average Precision score": APs, "True Negatives": CMs[:, 0, 0],
         "True Positives": CMs[:, 1, 1], "False Positives": CMs[:, 0, 1], "False Negatives": CMs[:, 1, 0],
         "Jaccard Score": jaccards, "F1 Score": F1s})
    metric_df.sort_values(by='Repeat', inplace=True)
    metric_df.to_csv(osp.join(base_dir, 'test_metrics.csv'), index=False)


def calculate_confusion_matrix(save_dir, classifier):
    experimental_path = osp.join(save_dir, 'experimental_df.csv')
    experimental_df = pd.read_csv(experimental_path)
    train = experimental_df[experimental_df['set'] == 'train']
    test = experimental_df[experimental_df['set'] == 'test']
    disp = plot_confusion_matrix(classifier, test[['x', 'y']], test[['labels']], display_labels=["Non Binder", "Binder"])
    plt.tight_layout()
    disp.ax_.tick_params(axis='both', which='major', labelsize='8')
    plt.savefig(osp.join(save_dir, 'results', 'classifier_CM_test.png'))
    plt.close()
    disp = plot_confusion_matrix(classifier, train[['x', 'y']], train[['labels']], display_labels=["Non Binder", "Binder"])
    plt.tight_layout()
    disp.ax_.tick_params(axis='both', which='major', labelsize='8')
    plt.savefig(osp.join(save_dir, 'results', 'classifier_CM_train.png'))
    plt.close()


def classify_SVC(args):
    experimental_path = osp.join(args.save_dir, 'experimental_df.csv')
    full_path = osp.join(args.save_dir, 'full_df.csv')
    experimental_df = pd.read_csv(experimental_path)
    train = experimental_df[experimental_df['set'] == 'train']
    test = experimental_df[experimental_df['set'] == 'test']
    # Fit classification model
    neigh = make_pipeline(StandardScaler(), SVC(gamma='auto', class_weight='balanced', probability=True))
    neigh.fit(train[['x', 'y']], train['labels'])
    # Save model to disk
    joblib.dump(neigh, osp.join(args.save_dir, 'model', 'svc_classifier.pkl'))
    test_predictions = neigh.predict(test[['x', 'y']])
    test_labels = test[['labels']]
    full_data = pd.read_csv(full_path)
    full_predictions = neigh.predict(full_data[['x', 'y']])
    h = 0.002
    x_min = min(train['x'].min() - .1, test['x'].min() - .1, full_data['x'].min() - .1)
    x_max = max(train['x'].max() + .1, test['x'].max() + .1, full_data['x'].max() + .1)
    y_min = min(train['y'].min() - .1, test['y'].min() - .1, full_data['y'].min() - .1)
    y_max = max(train['y'].max() + .1, test['y'].max() + .1, full_data['y'].max() + .1)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
    ax1.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r, alpha=0.1)
    ax1.scatter(train[train['labels'] == 1]['x'], train[train['labels'] == 1]['y'], c='b', alpha=0.1)
    ax1.scatter(train[train['labels'] == 0]['x'], train[train['labels'] == 0]['y'], c='r', alpha=0.1)
    ax1.scatter(test[test['labels'] == 1]['x'], test[test['labels'] == 1]['y'], c='b', marker='v', alpha=0.5)
    ax1.scatter(test[test['labels'] == 0]['x'], test[test['labels'] == 0]['y'], c='r', marker='v', alpha=0.5)
    ax1.set_xlabel('latent dimension 1')
    ax1.set_ylabel('latent dimension 2')
    ax1.set_title('D4 Experimentally Tested Compounds {0} Pair Model'.format(args.encoder_type))
    pop_a = mpatches.Patch(color='red', label='Non-binder')
    pop_b = mpatches.Patch(color='blue', label='Binder')
    shape_a = Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='black', markersize=15)
    shape_b = Line2D([0], [0], marker='v', color='w', label='Test', markerfacecolor='black', markersize=15)
    pop_c = mpatches.Patch(color='black', label='Full D4 Dataset')
    ax1.legend(handles=[pop_a, pop_b, shape_a, shape_b], loc='upper left')
    ax2.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r, alpha=0.1)
    ax2.scatter(full_data['x'], full_data['y'], alpha=0.025, s=100, c='black', marker='o', edgecolors='none')
    ax2.set_title('D4 Medium Diverse {0} Pair Model'.format(args.encoder_type))
    ax2.legend(handles=[pop_c], loc='upper left')
    ax1.set_xlabel('latent dimension 1')
    ax1.set_ylabel('latent dimension 2')
    ax2.set_xlabel('latent dimension 1')
    ax2.set_ylabel('latent dimension 2')
    matplotlib.rcParams.update({'font.size': 20})
    plt.rc('figure', titlesize=22)
    plt.savefig(osp.join(args.save_dir, 'results', 'svc_output.png'))
    return test_predictions, test_labels, full_predictions, neigh


def evaluate_pair_network_experimental(model, args, train_idx, val_idx):
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
    train_fp, test_fp = [], []
    with torch.no_grad():
        model.eval()
        for batch in train_support_dataloader:
            batch.to(args.device)
            fp = model(format_batch(args, batch)).detach().cpu().numpy()
            name = batch.name
            train_fp.append(fp)
            train_name.append(name)
        for batch in test_dataloader:
            batch.to(args.device)
            fp = model(format_batch(args, batch)).detach().cpu().numpy()
            name = batch.name
            test_fp.append(fp)
            test_name.append(name)
    train_fp = np.concatenate(train_fp)
    test_fp = np.concatenate(test_fp)
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
    fps = np.concatenate([train_fp, test_fp])
    labels = np.concatenate([train_activity_label, test_activity_label])
    group = np.concatenate([train_labels, test_labels])

    experimental_df = pd.DataFrame({'x': fps[:, 0], 'y': fps[:, 1], 'labels': labels, 'set': group, 'name': names})
    experimental_df.to_csv(osp.join(args.save_dir, 'experimental_df.csv'), index=False)


def evaluate_pair_network_full(model, args, full_dataset_path, outfile='full_df.csv'):
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
    full_fp = []
    with torch.no_grad():
        model.eval()
        for batch in full_dataloader:
            batch.to(args.device)
            fp = model(format_batch(args, batch)).detach().cpu().numpy()
            name = batch.name
            full_fp.append(fp)
            full_name.append(name)

    full_fp = np.concatenate(full_fp)
    name = np.concatenate(full_name)

    full_df = pd.DataFrame({'x': full_fp[:, 0], 'y': full_fp[:, 1], 'name': name})
    full_df.to_csv(osp.join(args.save_dir, outfile), index=False)


def load_args(checkpoint_path, dataset_path, savedir, device, epochs, fp_dim=256):
    """
    Loads arguments from checkpoint to initialize the pairnetwork
    :param checkpoint_path: The path of the directory which contains checkpoints. Should be a repeat dir i.e.
    checkpoint_path
    |->repeat_dir
      |->best_checkpoint.pt
    :param dataset_path: the path that contains the dataset to be evaluated using the pairnetwork.
    :param savedir: the path to save the trained model and all outputs.
    :param device: the device to train the model on.
    :param fp_dim: The size of the encoder output/FP
    :param epochs: Number of training epochs
    :return: An object of type TrainArgs for training
    """
    args = None
    stat_dict = None
    for subdir in os.listdir(checkpoint_path):
        current = osp.join(checkpoint_path, subdir)
        if os.path.isdir(current):
            checkpoint = osp.join(current, 'best_checkpoint.pt')
            model, args = load_checkpoint(checkpoint, device, return_args=True)
            stat_dict = model.state_dict()
            break
    full_dataset_path = args.data_path
    args.data_path = dataset_path
    args.save_dir = savedir
    if args.encoder_type == 'pfp':
        args.fp_dim = fp_dim
    args.batch_size = 128
    args.epochs = epochs

    return args, full_dataset_path, stat_dict


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
        for g1, g2, labels in train_loader:
            g1.to(args.device)
            g2.to(args.device)
            labels = labels.to(args.device)
            _, _, output = model(format_batch(args, g1), format_batch(args, g2))
            local_loss = criterion(output.squeeze(), labels)
            optimizer.zero_grad()
            local_loss.backward()
            optimizer.step()
            loss += local_loss.item()
        avg_train_loss = loss / len(train_loader)
        writer.add_scalar("Training loss", avg_train_loss, epoch+1)
        val_loss = 0.
        with torch.no_grad():
            model.eval()
            for g1, g2, labels in val_loader:
                g1.to(args.device)
                g2.to(args.device)
                labels = labels.to(args.device)
                _, _, output = model(format_batch(args, g1), format_batch(args, g2))
                local_loss = criterion(output.squeeze(), labels)
                val_loss += local_loss.item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = model.state_dict()
            save_checkpoint(osp.join(args.save_dir, 'best_checkpoint.pt'), model, args)
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Validation loss", avg_val_loss, epoch+1)
        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
            .format(epoch+1, args.epochs, avg_train_loss, avg_val_loss))
    model.load_state_dict(best_params)
    return model


class PFPSiameseEncoder(PFPEncoder):
    def __init__(self, args, node_dim, bond_dim):
        #TODO: Add funcionality to other encoder types
        super(PFPSiameseEncoder, self).__init__(args, node_dim, bond_dim)

    def forward(self, d1, d2=None):
        if d2 is not None:
            fp1, fp2 = super().forward(d1), super().forward(d2)
            return fp1, fp2
        else:
            fp1 = super().forward(d1)
            return fp1


class DMPNNSiameseEncoder(MPNEncoder):
    def __init__(self, args, node_dim, bond_dim):
        #TODO: Add funcionality to other encoder types
        super(DMPNNSiameseEncoder, self).__init__(args, node_dim, bond_dim)

    def forward(self, d1, d2=None):
        if d2 is not None:
            fp1, fp2 = super().forward(d1), super().forward(d2)
            return fp1, fp2
        else:
            fp1 = super().forward(d1)
            return fp1


class FPSiameseEncoder(FPEncoder):
    def __init__(self, args):
        super(FPSiameseEncoder, self).__init__(args)

    def forward(self, d1, d2=None):
        if d2 is not None:
            fp1, fp2 = super().forward(d1), super().forward(d2)
            return fp1, fp2
        else:
            fp1 = super().forward(d1)
            return fp1


class SiameseNetwork(nn.Module):
    def __init__(self, args):
        super(SiameseNetwork, self).__init__()
        self.args = args
        if self.args.encoder_type == 'fp':
            self.encoder = FPSiameseEncoder(args)
        else:
            if args.encoder_type == 'pfp':
                self.encoder = PFPSiameseEncoder(args, args.node_dim, args.edge_dim)
            elif args.encoder_type == 'dmpnn':
                self.encoder = DMPNNSiameseEncoder(args, args.node_dim, args.edge_dim)
            else:
                raise ValueError("Invalid encoder type for Siamese Network")
        self.fc_1 = nn.Linear(args.fp_dim, 128)
        self.fc_2 = nn.Linear(128, 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, d1, d2=None):
        if d2 is not None:
            fp1, fp2 = self.encoder.forward(d1, d2)
            fp1 = self.dropout(self.activation(self.fc_1(fp1)))
            fp2 = self.dropout(self.activation(self.fc_1(fp2)))
            fp1 = self.fc_2(fp1)
            fp2 = self.fc_2(fp2)
            distance = torch.sqrt(torch.sum(torch.square(fp1 - fp2), 1) + 1e-7)
            return fp1, fp2, distance
        else:
            fp1 = self.encoder.forward(d1)
            fp1 = self.dropout(self.activation(self.fc_1(fp1)))
            fp1 = self.fc_2(fp1)
            return fp1


class PairProximityGraphDataset(ProximityGraphDataset):
    def __init__(self, args, binding_file, set_size=16 * 100, val_percent=0.2, transform=None, pre_transform=None):
        self.binding_file = binding_file
        self.set_size = set_size
        self.val_percent = val_percent
        super(PairProximityGraphDataset, self).__init__(args, transform=transform, pre_transform=pre_transform)
        self.load_binding()
        self.generate_pairs()

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
        self.categories = {0: non_binder_index, 1: binder_index}

    def __len__(self):
        return self.set_size

    def getNodeDim(self):
        return self.data.x.numpy().shape[1]

    def getEdgeDim(self):
        return self.data.edge_attr.numpy().shape[1]

    def generate_pairs(self):
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
        self.train_binder_pairs = np.array(
            [(idx1, idx2) for i, idx1 in enumerate(train_binders) for idx2 in train_binders[i:]])
        self.train_non_binder_pairs = np.array(
            [(idx1, idx2) for i, idx1 in enumerate(train_non_binders) for idx2 in train_non_binders[i:]])
        self.train_negative_pairs = np.array(
            [(idx1, idx2) for i, idx1 in enumerate(train_non_binders) for idx2 in train_binders])
        self.val_binder_pairs = np.array([(idx1, idx2) for idx1 in train_binders for idx2 in val_binders])
        self.val_non_binder_pairs = np.array([(idx1, idx2) for idx1 in train_non_binders for idx2 in val_non_binders])
        self.val_negative_pairs = np.array([(idx1, idx2) for idx1 in train_binders for idx2 in val_non_binders] +
                                           [(idx1, idx2) for idx1 in train_non_binders for idx2 in val_binders])

        self.val_binders, self.val_non_binders, self.train_binders, self.train_non_binders = val_binders, val_non_binders, train_binders, train_non_binders

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, int):
            if idx % 2 == 0:
                category = np.random.choice([0, 1])
                if category == 0:
                    pair_idx = np.random.randint(
                        self.train_non_binder_pairs.shape[0]) if self.mode == 'train' else np.random.randint(
                        self.val_non_binder_pairs.shape[0])
                    idx_1, idx_2 = self.train_non_binder_pairs[pair_idx,
                                   :] if self.mode == 'train' else self.val_non_binder_pairs[pair_idx, :]
                else:
                    pair_idx = np.random.randint(
                        self.train_binder_pairs.shape[0]) if self.mode == 'train' else np.random.randint(
                        self.val_binder_pairs.shape[0])
                    idx_1, idx_2 = self.train_binder_pairs[pair_idx,
                                   :] if self.mode == 'train' else self.val_binder_pairs[pair_idx, :]
                data1 = self.get(idx_1)
                data2 = self.get(idx_2)
                data1 = data1 if self.transform is None else self.transform(data1)
                data2 = data2 if self.transform is None else self.transform(data2)
                label = 1.
            else:
                pair_idx = np.random.randint(
                    self.train_negative_pairs.shape[0]) if self.mode == 'train' else np.random.randint(
                    self.val_negative_pairs.shape[0])
                idx_1, idx_2 = self.train_negative_pairs[pair_idx,
                               :] if self.mode == 'train' else self.val_negative_pairs[pair_idx, :]
                data1 = self.get(idx_1)
                data2 = self.get(idx_2)
                data1 = data1 if self.transform is None else self.transform(data1)
                data2 = data2 if self.transform is None else self.transform(data2)
                label = 0.
            return data1, data2, torch.from_numpy(np.array(label, dtype=np.float32))
        else:
            return self.index_select(idx)

    def index_select(self, idx):
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                return self.index_select(idx.tolist())
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.index_select(
                    idx.nonzero(as_tuple=False).flatten().tolist())
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        dataset = copy.copy(self)
        dataset.__indices__ = indices
        categories = {0: [idx for idx in self.categories[0] if idx in indices],
                      1: [idx for idx in self.categories[1] if idx in indices]}
        dataset.categories = categories
        dataset.generate_pairs()
        return dataset


class PairFPDataset(FingerprintDataset):
    def __init__(self, args, binding_file, mode='train', set_size=16 * 100, val_percent=0.2, transform=None,
                 pre_transform=None):
        self.binding_file = binding_file
        self.set_size = set_size
        self.args = args
        self.val_percent = val_percent
        super(PairFPDataset, self).__init__(args, transform=transform, pre_transform=pre_transform)
        self.mode = mode
        self.load_binding()
        self.generate_pairs()
        self.set_of_idx1 = set()
        self.set_of_idx2 = set()

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
        self.categories = {0: non_binder_index, 1: binder_index}

    def generate_pairs(self):
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
        self.train_binder_pairs = np.array(
            [(idx1, idx2) for i, idx1 in enumerate(train_binders) for idx2 in train_binders[i:]])
        self.train_non_binder_pairs = np.array(
            [(idx1, idx2) for i, idx1 in enumerate(train_non_binders) for idx2 in train_non_binders[i:]])
        self.train_negative_pairs = np.array(
            [(idx1, idx2) for i, idx1 in enumerate(train_non_binders) for idx2 in train_binders])

        self.val_binder_pairs = np.array([(idx1, idx2) for idx1 in train_binders for idx2 in val_binders])
        self.val_non_binder_pairs = np.array([(idx1, idx2) for idx1 in train_non_binders for idx2 in val_non_binders])
        self.val_negative_pairs = np.array([(idx1, idx2) for idx1 in train_binders for idx2 in val_non_binders] +
                                           [(idx1, idx2) for idx1 in train_non_binders for idx2 in val_binders])

        self.val_binders, self.val_non_binders, self.train_binders, self.train_non_binders = val_binders, val_non_binders, train_binders, train_non_binders

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, int):
            if idx % 2 == 0:
                category = np.random.choice([0, 1])
                if category == 0:
                    if self.mode == 'train':
                        pair_idx = np.random.randint(self.train_non_binder_pairs.shape[0])
                        idx_1, idx_2 = self.train_non_binder_pairs[pair_idx, :]
                    else:
                        pair_idx = np.random.randint(self.val_non_binder_pairs.shape[0])
                        idx_1, idx_2 = self.val_non_binder_pairs[pair_idx, :]
                else:
                    if self.mode == 'train':
                        pair_idx = np.random.randint(self.train_binder_pairs.shape[0])
                        idx_1, idx_2 = self.train_binder_pairs[pair_idx, :]
                    else:
                        pair_idx = np.random.randint(self.val_binder_pairs.shape[0])
                        idx_1, idx_2 = self.val_binder_pairs[pair_idx, :]
                data1 = self.get(idx_1)
                data2 = self.get(idx_2)
                data1 = data1 if self.transform is None else self.transform(data1)
                data2 = data2 if self.transform is None else self.transform(data2)
                label = 1.
            else:
                if self.mode == 'train':
                    pair_idx = np.random.randint(self.train_negative_pairs.shape[0])
                    idx_1, idx_2 = self.train_negative_pairs[pair_idx, :]
                else:
                    pair_idx = np.random.randint(self.val_negative_pairs.shape[0])
                    idx_1, idx_2 = self.val_negative_pairs[pair_idx, :]
                data1 = self.get(idx_1)
                data2 = self.get(idx_2)

                data1 = data1 if self.transform is None else self.transform(data1)
                data2 = data2 if self.transform is None else self.transform(data2)
                label = 0.
            self.set_of_idx1.add(idx_1)
            self.set_of_idx2.add(idx_2)
            return data1, data2, torch.from_numpy(np.array(label, dtype=np.float32))
        else:
            return self.index_select(idx)

    def index_select(self, idx):
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                return self.index_select(idx.tolist())
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.index_select(
                    idx.nonzero(as_tuple=False).flatten().tolist())
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        dataset = copy.copy(self)
        dataset.__indices__ = indices
        categories = {0: [idx for idx in self.categories[0] if idx in indices],
                      1: [idx for idx in self.categories[1] if idx in indices]}
        dataset.categories = categories
        dataset.generate_pairs()
        return dataset


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):

        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))


        return loss