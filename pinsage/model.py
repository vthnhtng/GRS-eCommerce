import argparse
import os
import pickle

import dgl

import evaluation
import layers
import numpy as np
import sampler as sampler_module
import torch
import torch.nn as nn
import torchtext
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, cfg):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    timestamp = dataset["timestamp-edge-column"]

    # device = torch.device(cfg.device)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data["id"] = torch.arange(g.num_nodes(user_ntype))
    g.nodes[item_ntype].data["id"] = torch.arange(g.num_nodes(item_ntype))

    # Prepare torchtext dataset and Vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    for i in range(g.num_nodes(item_ntype)):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)
    for key, field in item_texts.items():
        vocab2 = build_vocab_from_iterator(
            textlist, specials=["<unk>", "<pad>"]
        )
        textset[key] = (
            textlist,
            vocab2,
            vocab2.get_stoi()["<pad>"],
            batch_first,
        )

    # Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, cfg.batch_size
    )
    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        cfg.random_walk_length,
        cfg.random_walk_restart_prob,
        cfg.num_random_walks,
        cfg.num_neighbors,
        cfg.num_layers,
    )
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset
    )
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=cfg.num_workers,
    )
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)),
        batch_size=cfg.batch_size,
        collate_fn=collator.collate_test,
        num_workers=cfg.num_workers,
    )
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, cfg.hidden_dims, cfg.num_layers
    ).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    start_epoch = 1
    # load existing model if exists
    if cfg.existing_model is not None:
        print("Loading existing model: {}...".format(
            cfg.existing_model))
        state = torch.load(
            os.path.join(cfg.model_dir, cfg.existing_model),
            map_location=device
        )
        model.load_state_dict(state['model_state_dict'])
        opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
    loss_list = []
    hit_list = []
    # For each batch of head-tail-negative triplets...
    for epoch_id in range(start_epoch, cfg.num_epochs + 1):
        print(f"Epoch {epoch_id}/{cfg.num_epochs}: ")
        batch_losses = []
        model.train()
        for batch_id in tqdm.trange(cfg.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            batch_losses.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        epoch_loss = np.mean(np.array(batch_losses))
        loss_list.append((epoch_id, epoch_loss))

        # Evaluate
        if (epoch_id % cfg.eval_freq == 0) or (epoch_id == cfg.num_epochs):
            model.eval()
            with torch.no_grad():
                item_batches = torch.arange(g.num_nodes(item_ntype)).split(
                    cfg.batch_size
                )
                h_item_batches = []
                for blocks in dataloader_test:
                    for i in range(len(blocks)):
                        blocks[i] = blocks[i].to(device)

                    h_item_batches.append(model.get_repr(blocks))
                h_item = torch.cat(h_item_batches, 0)
                hit_rate, _ = evaluation.evaluate_nn(
                    dataset, h_item, cfg.k, cfg.batch_size)
                hit_list.append((epoch_id, hit_rate))

                print(f"Validation (epoch {epoch_id}): loss: {epoch_loss}, hit@{cfg.k}: {hit_rate}")

        # Save model
        if (epoch_id % cfg.save_freq == 0) or (epoch_id == cfg.num_epochs):
            state = {
                'epoch': epoch_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss,
                'hit_rate': hit_rate, 
                'item_embeddings': h_item,
                'k': cfg.k,
                'batch_size': cfg.batch_size
            }

            model_fn = f"pinsage_{cfg.name}_{epoch_id}.pth"
            model_dir = cfg.model_dir
            torch.save(state, os.path.join(model_dir, model_fn))

    return model, h_item, loss_list, hit_list


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--random-walk-length", type=int, default=2)
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)
    parser.add_argument("--num-random-walks", type=int, default=10)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cpu"
    )  # can also be "cuda:0"
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    data_info_path = os.path.join(args.dataset_path, "data")
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, "train_g.bin")
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset["train-graph"] = g_list[0]
    train(dataset, args)
