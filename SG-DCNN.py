import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
import random
from tqdm import tqdm
import torch.nn.functional as F
import warnings


warnings.filterwarnings(
    'ignore',
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
    module='torch.serialization'
)


# ================== Self-Attention ==================
class SelfAttentionModule(nn.Module):
    def __init__(self, embed_size, heads=8):
        super(SelfAttentionModule, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, embed_size = x.shape
        x = x.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        out = attention_weights @ values
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.fc_out(out)


# ================== DCNN ==================
class DynamicCNN(nn.Module):
    def __init__(self, input_features, conv_config, attention_heads, num_hidden_layers):
        super(DynamicCNN, self).__init__()
        self.features = self._build_conv_layers(conv_config)
        self.classifier = self._build_classifier(input_features, conv_config, attention_heads, num_hidden_layers)

    def _build_conv_layers(self, config):
        layers = []
        in_channels = 1
        for layer in config:
            padding = (layer['kernel'] - 1) // 2 if layer.get('padding', 'same') == 'same' else 0
            layers += [
                nn.Conv1d(in_channels, layer['filters'], layer['kernel'], padding=padding),
                nn.ReLU(),
                nn.MaxPool1d(layer['pool']),
                nn.Dropout(layer['dropout'])
            ]
            in_channels = layer['filters']
        return nn.Sequential(*layers)

    def _build_classifier(self, input_features, conv_config, heads, num_hidden_layers):
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_features)
            conv_out = self.features(dummy).view(1, -1).shape[1]

        # Classifier construction
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Unflatten(1, (1, 64)),
            SelfAttentionModule(embed_size=64, heads=heads),
            nn.Flatten()
        ]
        last_hidden_size = 64

        hidden_sizes = [32, 16, 8]
        for i in range(num_hidden_layers):
            current_size = hidden_sizes[i]
            classifier_layers.extend([
                nn.Linear(last_hidden_size, current_size),
                nn.ReLU(),
                nn.Dropout(0.2 if i < num_hidden_layers - 1 else 0.1)
            ])
            last_hidden_size = current_size


        classifier_layers.append(nn.Linear(last_hidden_size, 2))
        return nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ================== 5 Cross-validation ==================
def kfold_evaluate(train_path, param_config, k=5, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        df = pd.read_csv(train_path)
        X = df.iloc[:, :-1].values.astype(np.float32).reshape(-1, 1, df.shape[1] - 1)
        y = df.iloc[:, -1].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y_encoded))
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=param_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=param_config['batch_size'], shuffle=False)

        input_features = X.shape[2]
        model = DynamicCNN(
            input_features,
            param_config['conv_config'],
            param_config['attention_heads'],
            param_config['num_hidden_layers']
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=param_config['lr'])
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        best_loss = float('inf')
        for epoch in range(50):
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            val_loss = epoch_loss / len(train_loader)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"fold_{fold}_best_model.pth")

        # Load the best model
        try:
            model.load_state_dict(torch.load(f"fold_{fold}_best_model.pth", map_location=device, weights_only=True),
                                  strict=True)
        except TypeError:
            model.load_state_dict(torch.load(f"fold_{fold}_best_model.pth", map_location=device), strict=True)

        metrics = evaluate_on_holdout(model, val_loader, device, le.classes_)  # 传递类别信息
        metrics_list.append(metrics)

    avg_metrics = {
        'mcc': np.mean([m['mcc'] for m in metrics_list]),
        'sn': np.mean([m['sn'] for m in metrics_list]),
        'sp': np.mean([m['sp'] for m in metrics_list]),
        'auc': np.mean([m['auc'] for m in metrics_list])
    }
    return avg_metrics


# ================== Evaluation index ==================
def evaluate_on_holdout(model, holdout_loader, device, classes=None):
    model.eval()
    all_probas = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels_batch in holdout_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.softmax(outputs, 1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_probas.extend(probas[:, 1] if len(classes) == 2 else probas)
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
    mcc = matthews_corrcoef(all_labels, all_preds)


    if len(classes) == 2:
        auc = roc_auc_score(all_labels, all_probas)  # 二分类直接使用
    else:

        auc = roc_auc_score(all_labels, all_probas, multi_class='ovr', average='macro')

    return {'auc': auc, 'mcc': mcc, 'sn': sensitivity, 'sp': specificity}


# ================== Training and evaluation ==================
def train_and_evaluate(train_path, holdout_path, output_dir, param_config,
                       batch_size=16, patience=3, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the training data
    train_df = pd.read_csv(train_path)
    X_train = train_df.iloc[:, :-1].values.astype(np.float32).reshape(-1, 1, train_df.shape[1] - 1)
    y_train = train_df.iloc[:, -1].values
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    full_train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_encoded))
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

    # 5 Cross-validation
    kfold_metrics = kfold_evaluate(train_path, param_config, k=5, device=device)
    if not kfold_metrics:
        return None

    # Load the independent test data and convert it using the training set encoder
    try:
        holdout_df = pd.read_csv(holdout_path)
        X_holdout = holdout_df.iloc[:, :-1].values.astype(np.float32).reshape(-1, 1, holdout_df.shape[1] - 1)
        y_holdout = le.transform(holdout_df.iloc[:, -1].values)  # Encoder conversion using the training set
        holdout_dataset = TensorDataset(torch.tensor(X_holdout), torch.tensor(y_holdout))
        holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"The independent test data loading failed: {str(e)}")
        return None

    # Retrain the model
    input_features = X_holdout.shape[2]
    model = DynamicCNN(
        input_features,
        param_config['conv_config'],
        param_config['attention_heads'],
        param_config['num_hidden_layers']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=param_config['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_loss = float('inf')
    for epoch in range(15):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in full_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        val_loss = epoch_loss / len(full_train_loader)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    # Load the best model
    try:
        model.load_state_dict(
            torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device, weights_only=True), strict=True)
    except TypeError:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device), strict=True)

    # Calculate the independent test indicators
    independent_metrics = evaluate_on_holdout(model, holdout_loader, device, le.classes_)

    return {
        'lr': param_config['lr'],
        'batch_size': param_config['batch_size'],
        'conv_config': str(param_config['conv_config']),
        'attention_heads': param_config['attention_heads'],
        'num_hidden_layers': param_config['num_hidden_layers'],
        'patience': param_config['patience'],
        'auc': independent_metrics['auc'],
        'mcc': independent_metrics['mcc'],
        'sn': independent_metrics['sn']
    }


# ================== Parameter search ==================
def random_search(param_space, train_path, holdout_path, num_trials=10):
    # Add the default convolution configuration
    if not param_space.get('conv_configs'):
        param_space['conv_configs'] = [
            [{'filters': 64, 'kernel': 4, 'pool': 2, 'dropout': 0.4, 'padding': 'same'},
             {'filters': 48, 'kernel': 3, 'pool': 2, 'dropout': 0.4, 'padding': 0},
             {'filters': 32, 'kernel': 2, 'pool': 2, 'dropout': 0.3, 'padding': 0}],
            [{'filters': 32, 'kernel': 3, 'pool': 2, 'dropout': 0.3, 'padding': 'same'},
             {'filters': 64, 'kernel': 3, 'pool': 2, 'dropout': 0.4, 'padding': 'same'},
             {'filters': 128, 'kernel': 2, 'pool': 2, 'dropout': 0.5, 'padding': 0}]
        ]
        print("Warning: When using the default convolution configuration, it is recommended to explicitly define conv_configs in param_space")

    search_results = []
    progress_bar = tqdm(range(num_trials), desc="Parameter search progress")

    for trial in progress_bar:
        params = {
            'lr': 10 ** random.uniform(-5, -3),
            'batch_size': random.choice([16, 32, 64]),
            'conv_config': random.choice(param_space['conv_configs']),
            'attention_heads': random.choice([4, 8]),
            'num_hidden_layers': random.randint(1, 3),
            'patience': random.choice([3, 5])
        }

        temp_dir = os.path.join(os.path.dirname(train_path), f"temp_trial_{trial}")
        os.makedirs(temp_dir, exist_ok=True)

        metrics = train_and_evaluate(
            train_path=train_path,
            holdout_path=holdout_path,
            output_dir=temp_dir,
            param_config=params,
            batch_size=params['batch_size'],
            patience=params['patience']
        )

        if metrics:
            search_results.append(metrics)
            progress_bar.set_postfix({
                'trial': trial + 1,
                'mcc': f"{metrics['mcc']:.4f}",
                'sn': f"{metrics['sn']:.4f}",
                'auc': f"{metrics['auc']:.4f}"
            })

        # Clear temporary documents
        for f in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)

    # Generate a DataFrame in the specified format
    search_df = pd.DataFrame(search_results)

    search_df = search_df[
        ['lr', 'batch_size', 'conv_config', 'attention_heads', 'num_hidden_layers', 'patience', 'auc', 'mcc', 'sn']]
    # Save as a CSV file
    csv_path = os.path.join(os.path.dirname(train_path), "search_results.csv")
    search_df.to_csv(csv_path, index=False)
    print(f"The result has been saved to: {csv_path}")
    return search_df


# ================== 主执行与绘图（含隐藏层图形） ==================
if __name__ == "__main__":

    train_path = "C:\\Users\\30321\\PycharmProjects\\pythonProject1\\数据集\\Po4-new.csv"
    holdout_path = "C:\\Users\\30321\\Desktop\\suangeng\\suan\\Po4\\Po4-du\\Po4-du.csv"

    output_root = os.path.join(os.path.dirname(train_path), "results")
    os.makedirs(output_root, exist_ok=True)


    param_space = {
        'conv_configs': [
            [{'filters': 8, 'kernel': 4, 'pool': 2, 'dropout': 0.4, 'padding': 'same'},
             {'filters': 8, 'kernel': 3, 'pool': 2, 'dropout': 0.4, 'padding': 0},
             {'filters': 8, 'kernel': 2, 'pool': 2, 'dropout': 0.3, 'padding': 0}],
            [{'filters': 4, 'kernel': 3, 'pool': 2, 'dropout': 0.3, 'padding': 'same'},
             {'filters': 4, 'kernel': 3, 'pool': 2, 'dropout': 0.4, 'padding': 'same'},
             {'filters': 4, 'kernel': 2, 'pool': 2, 'dropout': 0.5, 'padding': 0}]
        ]
    }

    search_df = random_search(param_space, train_path, holdout_path, num_trials=10)

    # Drawing: Only use independent test data
    plt.figure(figsize=(20, 8))

    # Batch Size
    plt.subplot(2, 2, 1)
    for bs in search_df['batch_size'].unique():
        subset = search_df[search_df['batch_size'] == bs]
        plt.scatter(subset['batch_size'], subset['mcc'], c='b', marker='o', label=f'MCC (BS={bs})')
        plt.scatter(subset['batch_size'], subset['sn'], c='r', marker='s', label=f'SN (BS={bs})')
        plt.scatter(subset['batch_size'], subset['auc'], c='g', marker='^', label=f'AUC (BS={bs})')
    plt.xlabel('Batch Size')
    plt.ylabel('Metrics (Independent Test)')
    plt.title('Batch Size vs Metrics (Independent Test)')
    plt.legend()

    # Learning Rate
    plt.subplot(2, 2, 2)
    plt.scatter(search_df['lr'], search_df['mcc'], c='b', marker='o', label='MCC')
    plt.scatter(search_df['lr'], search_df['sn'], c='r', marker='s', label='SN')
    plt.scatter(search_df['lr'], search_df['auc'], c='g', marker='^', label='AUC')
    plt.xlabel('Learning Rate')
    plt.ylabel('Metrics (Independent Test)')
    plt.title('Learning Rate vs Metrics (Independent Test)')
    plt.legend()

    # num_hidden_layers
    plt.subplot(2, 2, 3)
    for hl in search_df['num_hidden_layers'].unique():
        subset = search_df[search_df['num_hidden_layers'] == hl]
        plt.scatter(subset['num_hidden_layers'], subset['mcc'], c='b', marker='o', label=f'MCC (HL={hl})')
        plt.scatter(subset['num_hidden_layers'], subset['sn'], c='r', marker='s', label=f'SN (HL={hl})')
        plt.scatter(subset['num_hidden_layers'], subset['auc'], c='g', marker='^', label=f'AUC (HL={hl})')
    plt.xlabel('Hidden Layers')
    plt.ylabel('Metrics (Independent Test)')
    plt.title('Hidden Layers vs Metrics (Independent Test)')
    plt.xticks([1, 2, 3])
    plt.legend()

    # attention_heads
    plt.subplot(2, 2, 4)
    for ah in search_df['attention_heads'].unique():
        subset = search_df[search_df['attention_heads'] == ah]
        plt.scatter(subset['attention_heads'], subset['mcc'], c='b', marker='o', label=f'MCC (AH={ah})')
        plt.scatter(subset['attention_heads'], subset['sn'], c='r', marker='s', label=f'SN (AH={ah})')
        plt.scatter(subset['attention_heads'], subset['auc'], c='g', marker='^', label=f'AUC (AH={ah})')
    plt.xlabel('Attention Heads')
    plt.ylabel('Metrics (Independent Test)')
    plt.title('Attention Heads vs Metrics (Independent Test)')
    plt.xticks([4, 8])
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_root, "metrics_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Output the best result
    if not search_df.empty:
        best = search_df.sort_values('mcc', ascending=False).head(1)
        print("Optimal parameters and independent test indicators：")
        print(best[['num_hidden_layers', 'attention_heads', 'batch_size', 'lr', 'mcc', 'sn', 'auc']])
    else:
        print("The parameter search did not yield valid results")