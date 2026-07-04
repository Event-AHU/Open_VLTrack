import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import Vocab, UDEWTSentenceDataset, collate_fn
from backbone import BiaffineDependencyParser
from opt import train_loop, evaluate
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 函数加载 GloVe 嵌入
def load_glove_embeddings(glove_path, vocab, embed_dim):
    embeddings_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    
    # 初始化嵌入矩阵（使用随机初始化作为回退）
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab.word_to_id), embed_dim))
    
    # 填充已知词的 GloVe 向量
    for word, idx in vocab.word_to_id.items():
        if word.lower() in embeddings_dict and len(embeddings_dict[word.lower()]) == embed_dim:
            embedding_matrix[idx] = embeddings_dict[word.lower()]
        elif word in embeddings_dict and len(embeddings_dict[word]) == embed_dim:
            embedding_matrix[idx] = embeddings_dict[word]
    
    return torch.from_numpy(embedding_matrix).float()


def main():
    parser = argparse.ArgumentParser(description="Train DPModel2 dependency parser (UD EWT).")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--glove_path", type=str, required=True)
    parser.add_argument("--glove_dim", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--ffnn_dim", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=30)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--factor", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    ds = load_dataset("universal_dependencies", "en_ewt", cache_dir=args.cache_dir)
    vocab = Vocab(ds)

    train_dataset = UDEWTSentenceDataset(ds["train"], vocab)
    dev_dataset = UDEWTSentenceDataset(ds["validation"], vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    pre_trained_embeddings = load_glove_embeddings(args.glove_path, vocab, args.glove_dim)

    parser_model = BiaffineDependencyParser(
        vocab_size=len(vocab.word_to_id),
        upos_size=len(vocab.upos_to_id),
        embed_dim=args.glove_dim,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        ffnn_dim=args.ffnn_dim,
        deprel_size=len(vocab.deprel_to_id),
        pre_trained_embeddings=pre_trained_embeddings,
    ).to(device)

    optimizer = optim.Adam(parser_model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.factor,
        patience=args.patience,
        threshold=1e-4,
        min_lr=args.min_lr,
    )

    print(f"开始在 {device} 上训练...")
    best_uas = 0.0
    best_las = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss = train_loop(parser_model, train_loader, optimizer, device)
        dev_uas, dev_las = evaluate(parser_model, dev_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            latest_model_path = os.path.join(
                args.save_dir, f"latest_epoch_{epoch+1}_uas_{dev_uas:.4f}.pth"
            )
            torch.save(
                {
                    "model_state_dict": parser_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "dev_uas": dev_uas,
                    "dev_las": dev_las,
                },
                latest_model_path,
            )
            print(f"保存最新模型：{latest_model_path}")

        if dev_uas > best_uas:
            best_uas = dev_uas
            best_las = dev_las
            best_epoch = epoch + 1
            old_best_files = [f for f in os.listdir(args.save_dir) if f.startswith("best_model_")]
            for old_file in old_best_files:
                os.remove(os.path.join(args.save_dir, old_file))
            best_model_path = os.path.join(
                args.save_dir,
                f"best_model_epoch_{best_epoch}_uas_{best_uas:.4f}_las_{best_las:.4f}.pth",
            )
            torch.save(
                {
                    "model_state_dict": parser_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_epoch": best_epoch,
                    "best_uas": best_uas,
                    "best_las": best_las,
                },
                best_model_path,
            )
            print(f"发现最优模型：{best_model_path}")

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Dev UAS: {dev_uas:.4f}, Dev LAS: {dev_las:.4f}, "
            f"Current LR: {current_lr:.6f}, "
            f"Best UAS: {best_uas:.4f} (Epoch {best_epoch})",
            flush=True,
        )

        scheduler.step(dev_uas)

    final_model_path = os.path.join(
        args.save_dir, f"final_model_epoch_{args.epochs}_uas_{dev_uas:.4f}.pth"
    )
    torch.save(
        {
            "model_state_dict": parser_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "final_epoch": args.epochs,
            "final_uas": dev_uas,
            "final_las": dev_las,
            "best_uas": best_uas,
            "best_las": best_las,
        },
        final_model_path,
    )
    print(f"训练结束，最终模型：{final_model_path}")
    print(f"最优模型：UAS {best_uas:.4f}, LAS {best_las:.4f}（Epoch {best_epoch}）")


if __name__ == "__main__":
    main()
