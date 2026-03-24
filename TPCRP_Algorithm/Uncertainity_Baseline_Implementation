import matplotlib.pyplot as plt 
import numpy as np 

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset 


import torchvision 
import torchvision.transforms as transforms 

from sklearn.cluster import KMeans 
from sklearn.neighbors import NearestNeighbors

from typing import List, Tuple 


def get_initial_seed(train_dataset, n_per_class=1):
    labels = np.array(train_dataset.targets)
    seed = []
    for c in range(10): 
        idx = np.where(labels == c)[0]
        seed.append(int(np.random.choice(idx)))
    return seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    ),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=base_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=base_transform
)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
budgets = [10, 20, 40, 80]
cumulative_budgets = [10,30,70,150] # For cumulative values 

# Reduced to make computation more efficient 
class SimpleClassifier(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(8*8*128, 256), nn.ReLU(),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

    

def score_least_confidence(logits):
    probs = torch.softmax(logits ,  dim=1)
    return ( 1- probs.max(dim=1).values)

def score_margin(logits):
    probs = torch.softmax(logits,dim=1)
    top2 = torch.topk(probs, 2, dim=1).values

    return top2[:,0] - top2[:,1] 


def score_entropy(logits):
    probs = torch.softmax(logits,dim=1)
    return - 1 * ( probs * probs.log()).sum(dim=1)



def compute_badge_embeddings(model, loader, device):
    model.eval()
    grads = [] 



    for x, _ in loader:
        x = x.to(device)
        x.requires_grad=True

        logits =model(x)
        probs = torch.softmax(logits, dim=1)

        y_hat = probs.argmax(dim=1)



        loss = F.cross_entropy(logits, y_hat, reduction='sum')
        loss.backward()


        grads.append(x.grad.view(x.size(0), -1).cpu().numpy())
    return np.concatenate(grads, axis=0)

def select_by_uncertainty(model, unlabeled_loader, device, B, scoring_fn):
    model.eval()
    scores = []

    with torch.no_grad():
        for x, idx in unlabeled_loader:
            x=x.to(device)
            logits = model(x)
            s =  scoring_fn(logits)
            scores.extend(list(zip(idx.numpy(), s.cpu(). numpy()) ))

    # desc. uncertainity
    scores = sorted(scores, key=lambda x: -x[1])
    selected = [idx for idx, _ in scores[:B]]

    return selected 


def plot_baseline_comparison(results_dict, budgets):
    cumulative_budgets = np.cumsum(budgets).tolist()

    plt.figure(figsize=(8,5))

    for method, accs in results_dict.items():
        if accs is None or len(accs) == 0:
            continue
        if not all(isinstance(a, (int, float, np.floating)) for a in accs if a is not None):
            continue

        plt.plot(cumulative_budgets, accs, marker="x", label=method)

    plt.xlabel("Cumulative Budget")
    plt.ylabel("Accuracy")
    plt.title("Active Learning Baseline Comparison (Cumulative Budget)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("baseline_comparison_cumulative.png", dpi=300)
    plt.show()




def evaluate(model, loader, device): 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)

            preds = logits.argmax(dim=1)
            correct += (preds ==y ).sum().item()

            total += y.size(0)
        
        return correct / total
    
def active_learning_round(
    model,
    train_dataset,
    test_loader,
    labeled_indices,
    unlabeled_indices,
    B,
    scoring_fn,
    device
):
    labeled_subset = torch.utils.data.Subset(train_dataset, labeled_indices)
    loader = DataLoader(labeled_subset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):  
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    unlabeled_subset = torch.utils.data.Subset(train_dataset, unlabeled_indices)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=128, shuffle=False)

    selected = select_by_uncertainty(model, unlabeled_loader, device, B, scoring_fn)
    selected_global = [unlabeled_indices[i] for i in selected]

    new_labeled = labeled_indices + selected_global
    new_unlabeled = [i for i in unlabeled_indices if i not in selected_global]

    acc = evaluate(model, test_loader, device)

    return new_labeled, new_unlabeled, acc


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, x, T=3):
    #model.train()  # keep dropout ON
    enable_dropout(model)
    preds = []
    for _ in range(T):
        logits = model(x)
        preds.append(torch.softmax(logits, dim=1).unsqueeze(0))
    return torch.cat(preds, dim=0)  # [t, b,   c]

# BALD method 
def score_bald_from_mc(mc_probs):
    mean_probs = mc_probs.mean(dim=0)
    entropy_mean = -(mean_probs * mean_probs.log()).sum(dim=1)
    entropy_expected = -(mc_probs * mc_probs.log()).sum(dim=2).mean(dim=0)
    return entropy_mean - entropy_expected


def score_bald(model, x, device, T=3):
    x = x.to(device)
    mc_probs = mc_dropout_predict(model, x, T=T)
    return score_bald_from_mc(mc_probs)

def score_dbal_from_mc(mc_probs):
    preds = mc_probs.argmax(dim=2)  # [T, B]
    mode_counts = torch.mode(preds, dim=0).values
    return 1 - (mode_counts / mc_probs.size(0))


def select_badge(model, unlabeled_loader, device, B):
    model.eval()

    # Collect embeddings AND global indices in the same order
    all_embeddings = []
    all_indices = []

    for x, idx in unlabeled_loader:
        x = x.to(device)
        x.requires_grad = True

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        y_hat = probs.argmax(dim=1)

        loss = F.cross_entropy(logits, y_hat, reduction='sum')
        model.zero_grad()
        loss.backward()

        grads = x.grad.view(x.size(0), -1).detach().cpu().numpy()
        all_embeddings.append(grads)
        all_indices.extend(idx.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # KMeans++ selection
    kmeans = KMeans(n_clusters=B, init="k-means++").fit(all_embeddings)
    centers = kmeans.cluster_centers_

    # Compute distances to centers
    dists = np.linalg.norm(all_embeddings[:, None] - centers[None, :], axis=2)
    chosen = np.argmin(dists, axis=0)

    return [all_indices[i] for i in chosen]



def select_random(unlabeled_indices, B):
    return list(np.random.choice(unlabeled_indices, size=B, replace=False))

def evaluate_fixed_selection(model, train_dataset, test_loader, indices, device):
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return evaluate(model, test_loader, device)


def run_all_baselines(train_dataset, test_loader, budgets, device):
    print("\n=== Starting all baselines ===")

    results = {
        "Least Confidence": [],
        "Margin": [],
        "Entropy": [],
        "BADGE": [],
        "BALD": [],
        "DBAL": [],
        "Random": [],
        "TPC-RP": [],
        "TPC-DC": []  
    }

    scoring_fns = {
        "Least Confidence": score_least_confidence,
        "Margin": score_margin,
        "Entropy": score_entropy
    }


    for method, fn in scoring_fns.items():
        print(f"\n=== Running baseline: {method} ===")
        labeled = get_initial_seed(train_dataset)
        unlabeled = list(range(len(train_dataset)))
        accs = []

        for B in budgets:
            print(f"  -> Budget {B}: training classifier...")
            model = SimpleClassifier().to(device)

            labeled, unlabeled, acc = active_learning_round(
                model, train_dataset, test_loader,
                labeled, unlabeled, B,
                fn, device
            )

            print(f"     Accuracy after budget {B}: {acc:.4f}")
            accs.append(acc)

        results[method] = accs

    """
    print("\n=== Running baseline: BADGE ===")
    labeled = get_initial_seed(train_dataset)
    unlabeled = list(range(len(train_dataset)))
    accs = []

    for B in budgets:
        print(f"  -> BADGE selecting {B} points...")
        model = SimpleClassifier().to(device)

        unlabeled_subset = Subset(train_dataset, unlabeled)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=128, shuffle=False)

        selected = select_badge(model, unlabeled_loader, device, B)
        labeled += selected
        unlabeled = [i for i in unlabeled if i not in selected]

        acc = evaluate(model, test_loader, device)
        print(f"     Accuracy after BADGE budget {B}: {acc:.4f}")
        accs.append(acc)

    results["BADGE"] = accs
    """


    print("\n=== Running baseline: BALD ===")
    labeled = get_initial_seed(train_dataset)
    unlabeled = list(range(len(train_dataset)))
    accs = []

    for B in budgets:
        print(f"  -> BALD scoring unlabeled pool for budget {B}...")
        model = SimpleClassifier(dropout=True).to(device)

        unlabeled_subset = Subset(train_dataset, unlabeled)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=128, shuffle=False)

        selected = []
        for x, idx in unlabeled_loader:
            scores = score_bald(model, x, device)
            selected.extend(list(zip(idx.numpy(), scores.detach().cpu().numpy())))

        selected = sorted(selected, key=lambda x: -x[1])[:B]
        selected = [i for i, _ in selected]

        labeled += selected
        unlabeled = [i for i in unlabeled if i not in selected]

        acc = evaluate(model, test_loader, device)
        print(f"     Accuracy after BALD budget {B}: {acc:.4f}")
        accs.append(acc)

    results["BALD"] = accs



    print("\n=== Running baseline: DBAL ===")
    labeled = get_initial_seed(train_dataset)
    unlabeled = list(range(len(train_dataset)))
    accs = []

    for B in budgets:
        print(f"  -> DBAL scoring unlabeled pool for budget {B}...")
        model = SimpleClassifier(dropout=True).to(device)

        unlabeled_subset = Subset(train_dataset, unlabeled)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=128, shuffle=False)

        selected = []
        for x, idx in unlabeled_loader:
            mc_probs = mc_dropout_predict(model, x.to(device))
            scores = score_dbal_from_mc(mc_probs)
            selected.extend(list(zip(idx.numpy(), scores.detach().cpu().numpy())))

        selected = sorted(selected, key=lambda x: -x[1])[:B]
        selected = [i for i, _ in selected]

        labeled += selected
        unlabeled = [i for i in unlabeled if i not in selected]

        acc = evaluate(model, test_loader, device)
        print(f"     Accuracy after DBAL budget {B}: {acc:.4f}")
        accs.append(acc)

    results["DBAL"] = accs
    
    print("\n=== Evaluating TPC-RP selections ===")
    for B in budgets:
        print(f"  -> Evaluating TPC-RP for budget {B}...")

        # local path 
        indices = np.load(f"/home/ariag/5CCSAMLF/large_meshupar/TPCRP_Algorithm/unmodified_budget_results/typiclust_B{B}.npy")
        model = SimpleClassifier().to(device)
        acc = evaluate_fixed_selection(model, train_dataset, test_loader, indices, device)
        print(f"     TPC-RP accuracy for budget {B}: {acc:.4f}")
        results["TPC-RP"].append(acc)
    print("\n=== Evaluating TPC-DC selections ===")
    for B in budgets:
        print(f"  -> Evaluating TPC-DC for budget {B}...")
        try:
            indices = np.load(f"tpcdc_B{B}.npy")
            model = SimpleClassifier().to(device)
            acc = evaluate_fixed_selection(model, train_dataset, test_loader, indices, device)
            print(f"     TPC-DC accuracy for budget {B}: {acc:.4f}")
            #results["TPC-DC"].append(acc)
            results["TPC-DC"].append(None)
        except FileNotFoundError:
            print(f"     No TPC-DC file found for budget {B}.")
            results["TPC-DC"].append(None)

    while len(results["TPC-DC"]) < len(budgets):
        results["TPC-DC"].append(None)


    print("\n=== All baselines complete ===")
    return results


results = run_all_baselines(train_dataset, test_loader, budgets, device)
plot_baseline_comparison(results, budgets)