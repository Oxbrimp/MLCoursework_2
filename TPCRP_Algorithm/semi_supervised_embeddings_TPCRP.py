import os
import numpy as np
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

import hdbscan
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUDGETS = [10, 20, 40, 80]
SSL_EPOCHS = 200
BATCH_SIZE_SSL = 256
BATCH_SIZE_LABELED = 32
BATCH_SIZE_UNLABELED = 128
NUM_CLASSES = 10
MC_DROPOUT_SAMPLES = 20
TSNE_PERPLEXITY = 50
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


class ResNetEncd(nn.Module):
    def __init__(self, base="resnet18", proj_dim=128):
        super().__init__()
        backbone = getattr(torchvision.models, base)(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.feat_dim = feat_dim
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim)
        )

    def forward(self, x, return_projection: bool = True):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        if return_projection:
            proj = self.projection(feat)
            return feat, proj
        return feat

    def represent(self, x):
        return self.forward(x, return_projection=False)




class ContrastiveNTXent(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        batch = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        diag_mask = torch.eye(2 * batch, device=z.device).bool()
        sim_matrix.masked_fill_(diag_mask, -9e15)
        labels = torch.arange(batch, device=z.device)
        labels = torch.cat([labels + batch, labels], dim=0)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class TwoCropTransform:
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

    def __call__(self, img):
        return self.weak(img), self.weak(img)  # for SimCLR training

    def weak_transform(self, img):
        return self.weak(img)

    def strong_transform(self, img):
        return self.strong(img)



def compute_typicality(features: np.ndarray, k: int = 20) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(features)), algorithm='auto').fit(features)
    distances, _ = nbrs.kneighbors(features)
    distances = distances[:, 1:]
    avg_sq_dist = (distances ** 2).mean(axis=1)
    typicality = 1.0 / (avg_sq_dist + 1e-12)
    return typicality


def distance_to_selected(indices, features, labeled):
    if len(labeled) == 0:
        return np.zeros(len(indices))
    selected_feats = features[list(labeled)]
    dists = np.linalg.norm(features[indices][:, None] - selected_feats[None, :], axis=2)
    return dists.min(axis=1)


def typiclust_hdbscan_selection(features: np.ndarray,
                                budget_total: int,
                                k_typicality: int = 20,
                                lambda_: float = 0.01,
                                min_cluster_size: int = 30,
                                min_samples: int = 5,
                                random_state: int = 0) -> List[int]:

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric="euclidean")
    cluster_labels = clusterer.fit_predict(features)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    n_noise = int(np.sum(cluster_labels == -1))
    n_clusters = int((unique != -1).sum())
    print(f"[HDBSCAN] clusters={n_clusters}, noise={n_noise}, total={len(cluster_labels)}")

    N = features.shape[0]
    labeled = set()
    unlabeled = set(range(N))
    typicality = compute_typicality(features, k=k_typicality)

    valid_mask = cluster_labels != -1
    unique_clusters = np.unique(cluster_labels[valid_mask])

    while len(labeled) < budget_total:
        uncovered = [c for c in unique_clusters if not any(cluster_labels[i] == c for i in labeled)]
        if len(uncovered) == 0:
            break
        new_indices = []
        for c in uncovered:
            if len(labeled) + len(new_indices) >= budget_total:
                break
            cluster_idx = np.where(cluster_labels == c)[0]
            cluster_idx = [i for i in cluster_idx if i in unlabeled]
            if not cluster_idx:
                continue
            scores = typicality[cluster_idx] - lambda_ * distance_to_selected(cluster_idx, features, labeled)
            best_local = cluster_idx[int(np.argmax(scores))]
            new_indices.append(best_local)
        if not new_indices:
            break
        for idx in new_indices:
            labeled.add(idx)
            unlabeled.remove(idx)
    return sorted(labeled)

class LabeledDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], transform=None):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class UnlabeledDataset(Dataset):
    def __init__(self, base_dataset: Dataset, labeled_indices: List[int], weak_transform, strong_transform):
        self.base = base_dataset
        self.labeled_set = set(labeled_indices)
        self.indices = [i for i in range(len(base_dataset)) if i not in self.labeled_set]
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, _ = self.base[idx]
        x_w = self.weak_transform(x)
        x_s = self.strong_transform(x)
        return x_w, x_s


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)



class FlexMatchLite:

    def __init__(self, encoder: nn.Module, feat_dim: int, num_classes: int = 10, device=DEVICE, head_dropout: float = 0.0):
        self.encoder = encoder
        self.encoder.eval()  # encoder frozen
        self.head = LinearHead(in_dim=feat_dim, num_classes=num_classes, dropout=head_dropout).to(device)
        self.device = device
        self.num_classes = num_classes
        self.class_thresholds = np.full(num_classes, 0.95, dtype=float)
        self.momentum = 0.9

    def train(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader,
              epochs: int = 50, tau: float = 0.95, lambda_u: float = 1.0, lr: float = 1e-3):
        opt = torch.optim.Adam(self.head.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()
        unlabeled_iter = iter(unlabeled_loader)

        for epoch in range(epochs):
            running_sup = 0.0
            running_unsup = 0.0
            n_sup = 0
            n_unsup = 0

            for x_l, y_l in labeled_loader:
                # supervised batch
                x_l = x_l.to(self.device)
                y_l = y_l.to(self.device)
                with torch.no_grad():
                    h_l = self.encoder.represent(x_l)
                logits_l = self.head(h_l)
                loss_sup = ce(logits_l, y_l)

                # unlabeled batch
                try:
                    x_w, x_s = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    x_w, x_s = next(unlabeled_iter)

                x_w = x_w.to(self.device)
                x_s = x_s.to(self.device)

                with torch.no_grad():
                    h_w = self.encoder.represent(x_w)
                    logits_w = self.head(h_w)
                    probs_w = F.softmax(logits_w, dim=1)
                    confs, pseudo = probs_w.max(dim=1)

                thresholds = torch.tensor(self.class_thresholds, device=self.device, dtype=confs.dtype)
                mask = confs >= thresholds[pseudo]

                if mask.sum() > 0:
                    with torch.no_grad():
                        h_s = self.encoder.represent(x_s)
                    logits_s = self.head(h_s)
                    loss_unsup = ce(logits_s[mask], pseudo[mask])
                else:
                    loss_unsup = torch.tensor(0.0, device=self.device)

                loss = loss_sup + lambda_u * loss_unsup

                opt.zero_grad()
                loss.backward()
                opt.step()

                probs_np = probs_w.cpu().numpy()
                preds_np = pseudo.cpu().numpy()
                for c in range(self.num_classes):
                    sel = (preds_np == c)
                    if sel.sum() > 0:
                        mean_conf = probs_np[sel, c].mean()
                        self.class_thresholds[c] = self.momentum * self.class_thresholds[c] + (1 - self.momentum) * mean_conf

                running_sup += loss_sup.item() * x_l.size(0)
                running_unsup += (loss_unsup.item() if isinstance(loss_unsup, torch.Tensor) else 0.0) * x_l.size(0)
                n_sup += x_l.size(0)
                n_unsup += x_l.size(0)

            avg_sup = running_sup / max(1, n_sup)
            avg_unsup = running_unsup / max(1, n_unsup)
            print(f"[FlexMatch] Epoch {epoch+1}/{epochs} sup={avg_sup:.4f} unsup={avg_unsup:.4f}")

        return self.head



def compute_entropy(probs: np.ndarray) -> np.ndarray:
    ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    return ent


def compute_margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return margin


def compute_bald_mc_dropout(model_head: nn.Module, encoder: nn.Module, dataloader: DataLoader,
                           mc_samples: int = 20, device=DEVICE) -> np.ndarray:
    model_head.train()  # enable dropout
    encoder.eval()
    all_probs = []
    with torch.no_grad():
        for _ in range(mc_samples):
            probs_list = []
            for x, _ in dataloader:
                x = x.to(device)
                h = encoder.represent(x)
                logits = model_head(h)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
            all_probs.append(np.vstack(probs_list))
    all_probs = np.stack(all_probs, axis=0)  # [mc, N, C]
    mean_probs = all_probs.mean(axis=0)
    entropy_mean = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-12, 1.0)), axis=1)
    mean_entropy = -np.sum(all_probs * np.log(np.clip(all_probs, 1e-12, 1.0)), axis=2).mean(axis=0)
    bald = entropy_mean - mean_entropy
    model_head.eval()
    return bald


def linear_evaluation(encoder: nn.Module, head: nn.Module, test_loader: DataLoader, device=DEVICE) -> float:
    encoder.eval()
    head.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            h = encoder.represent(x)
            logits = head(h)
            pred = logits.argmax(dim=1)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = accuracy_score(trues, preds)
    return acc


def tsne_and_plot(features: np.ndarray, selected_indices: List[int], uncertainties: np.ndarray = None,
                  title: str = "t-SNE", save_path: str = None):
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=SEED)
    proj = tsne.fit_transform(features)
    plt.figure(figsize=(8, 8))
    plt.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.4, color="tab:gray")
    if len(selected_indices) > 0:
        sel = np.array(selected_indices)
        sel = sel[sel < proj.shape[0]]  # safety
        plt.scatter(proj[sel, 0], proj[sel, 1], s=60, marker="x", color="red", label="selected")
    if uncertainties is not None:
        u = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-12)
        plt.scatter(proj[:, 0], proj[:, 1], c=u, cmap="viridis", s=8, alpha=0.6)
        plt.colorbar(label="uncertainty")
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def generate_typiclust_for_budget(features, B, save_dir):
    path = os.path.join(save_dir, f"typiclust_HDBSCAN_B{B}.npy")
    if os.path.exists(path):
        print(f"Already exists: {path}")
        return
    print(f"Generating TypiClust for B={B}")
    idx = typiclust_hdbscan_selection(features, budget_total=B)
    np.save(path, np.array(idx))
    print("Saved:", path)





def run_semi_supervised_pipeline(
        data_root: str = "./data",
        budgets: List[int] = BUDGETS,

        ssl_epochs: int = SSL_EPOCHS,

        use_pretrained_ssl_checkpoint: str = None,
        save_dir: str = "semi_supervised_results"
):
    os.makedirs(save_dir, exist_ok=True)

        # transforms and datasets (single, correct block)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # dataset that returns normalized tensors for supervised training/eval
    cifar_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=base_transform)
    cifar_test  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=base_transform)

    # dataset that returns raw PIL images for SSL / unlabeled transforms
    raw_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)

    ssl_transform = TwoCropTransform()

    # define TwoCropWrapper BEFORE using it
    class TwoCropWrapper(Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, _ = self.base[i]   # raw PIL image
            x1, x2 = self.transform(img)
            return x1, x2, i

    # wrap the raw dataset for SSL training
    ssl_wrapped = TwoCropWrapper(raw_train, ssl_transform)




    encoder = ResNetEncd(base="resnet18", proj_dim=128).to(DEVICE)

    if use_pretrained_ssl_checkpoint and os.path.exists(use_pretrained_ssl_checkpoint):
        print("Loading SSL checkpoint:", use_pretrained_ssl_checkpoint)
        encoder.load_state_dict(torch.load(use_pretrained_ssl_checkpoint, map_location=DEVICE))
    else:
        print("Training SSL encoder (SimCLR) for", ssl_epochs, "epochs")
        encoder.train()
        loader = DataLoader(ssl_wrapped, batch_size=BATCH_SIZE_SSL, shuffle=True, num_workers=4, drop_last=True)
        loss_fn = ContrastiveNTXent(temperature=0.2).to(DEVICE)
        opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        for epoch in range(ssl_epochs):
            running = 0.0
            for x1, x2, _ in loader:
                x1 = x1.to(DEVICE); x2 = x2.to(DEVICE)
                _, z1 = encoder(x1, return_projection=True)
                _, z2 = encoder(x2, return_projection=True)
                loss = loss_fn(z1, z2)
                opt.zero_grad(); loss.backward(); opt.step()
                running += loss.item() * x1.size(0)
            print(f"[SSL] Epoch {epoch+1}/{ssl_epochs} loss={running/len(loader.dataset):.4f}")
        torch.save(encoder.state_dict(), os.path.join(save_dir, "ssl_encoder.pth"))

    print("Extracting features for all train images")
    encoder.eval()

    train_loader = DataLoader(cifar_train, batch_size=256, shuffle=False, num_workers=4)
    all_feats = []
    with torch.no_grad():
        for x, _ in tqdm(train_loader):
            x = x.to(DEVICE)
            feats = encoder.represent(x)
            all_feats.append(feats.cpu().numpy())
    features = np.concatenate(all_feats, axis=0)  # shape [N, feat_dim]


    for B in budgets:
        print("\n" + "="*40)
        print(f"Budget B={B}")

        # --- RESUME LOGIC ---
        selection_path = os.path.join(save_dir, f"typiclust_HDBSCAN_B{B}.npy")
        if os.path.exists(selection_path):
            print(f"Found existing selection for B={B}, loading...")
            selected = np.load(selection_path)
        else:
            print(f"No selection found for B={B}, running TypiClust...")
            selected = typiclust_hdbscan_selection(
                features, budget_total=B, k_typicality=20, lambda_=0.01
            )
            np.save(selection_path, np.array(selected))
        # ---------------------

        print(f"Selected {len(selected)} indices")

        # Build datasets
        labeled_ds = LabeledDataset(cifar_train, selected, transform=None)   # keep normalized tensors for supervised head
        unlabeled_ds = UnlabeledDataset(raw_train, selected, weak_transform=ssl_transform.weak_transform, strong_transform=ssl_transform.strong_transform)


        labeled_loader = DataLoader(
            labeled_ds, batch_size=BATCH_SIZE_LABELED,
            shuffle=True, num_workers=4
        )
        unlabeled_loader = DataLoader(
            unlabeled_ds, batch_size=BATCH_SIZE_UNLABELED,
            shuffle=True, num_workers=4
        )

        # FlexMatch training
        fm = FlexMatchLite(
            encoder=encoder,
            feat_dim=encoder.feat_dim,
            num_classes=NUM_CLASSES,
            device=DEVICE,
            head_dropout=0.0
        )
        head = fm.train(
            labeled_loader, unlabeled_loader,
            epochs=30, tau=0.95, lambda_u=1.0, lr=1e-3
        )

        torch.save(
            head.state_dict(),
            os.path.join(save_dir, f"flexmatch_head_B{B}.pth")
        )


        test_loader = DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=4)
        acc = linear_evaluation(encoder, head, test_loader, device=DEVICE)
        print(f"[RESULT] FlexMatch-lite test accuracy (B={B}): {acc:.4f}")

        head.eval()
        probs_all = []
        train_loader_eval = DataLoader(cifar_train, batch_size=256, shuffle=False, num_workers=4)
        with torch.no_grad():
            for x, _ in train_loader_eval:
                x = x.to(DEVICE)
                h = encoder.represent(x)
                logits = head(h)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_all.append(probs)
        probs_all = np.vstack(probs_all)
        entropy_scores = compute_entropy(probs_all)
        margin_scores = compute_margin(probs_all)

        dropout_head = LinearHead(in_dim=encoder.feat_dim, num_classes=NUM_CLASSES, dropout=0.5).to(DEVICE)
        dropout_head.load_state_dict(head.state_dict())  # copy weights
        dropout_head.train()
        opt_ft = torch.optim.Adam(dropout_head.parameters(), lr=1e-3)



        for _ in range(2):
            for x_l, y_l in labeled_loader:
                x_l = x_l.to(DEVICE); y_l = y_l.to(DEVICE)
                with torch.no_grad():
                    h_l = encoder.represent(x_l)
                logits_ft = dropout_head(h_l)
                loss_ft = F.cross_entropy(logits_ft, y_l)
                opt_ft.zero_grad(); loss_ft.backward(); opt_ft.step()
        dropout_head.eval()

        try:
            bald_scores = compute_bald_mc_dropout(dropout_head, encoder, train_loader_eval, mc_samples=MC_DROPOUT_SAMPLES, device=DEVICE)
            np.save(os.path.join(save_dir, f"bald_B{B}.npy"), bald_scores)
            print("Saved BALD scores.")
        except Exception as e:
            print("BALD computation failed. Error:", e)
            bald_scores = None

        tsne_and_plot(features, selected, uncertainties=entropy_scores,
                      title=f"t-SNE (B={B}) - entropy overlay",
                      save_path=os.path.join(save_dir, f"tsne_entropy_B{B}.png"))

        np.save(os.path.join(save_dir, f"entropy_B{B}.npy"), entropy_scores)
        np.save(os.path.join(save_dir, f"margin_B{B}.npy"), margin_scores)
        if bald_scores is not None:
            np.save(os.path.join(save_dir, f"bald_B{B}.npy"), bald_scores)

    print("Pipeline complete.")


if __name__ == "__main__":
    features = np.load("../")
    generate_typiclust_for_budget(features, 20, "semi_supervised_results")
    generate_typiclust_for_budget(features, 40, "semi_supervised_results")
    generate_typiclust_for_budget(features, 80, "semi_supervised_results")

    """
    run_semi_supervised_pipeline(data_root="./data",
                                budgets=[10, 20, 40, 80],
                                ssl_epochs=500,  
                                # Continuation point 
                                use_pretrained_ssl_checkpoint="/home/ariag/5CCSAMLF/large_meshupar/budget_results/semi_supervised_budget_Results/ssl_encoder.pth",
                                save_dir="semi_supervised_results")
    """
