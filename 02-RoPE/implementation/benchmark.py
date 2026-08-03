from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import baseline
import roformer


def make_reverse_pairs(seq_len, num_pairs, num_embeddings, seed):
    # tgt[i] = src[seq_len-1-i] -- the required offset changes with absolute
    # position i, so this rewards absolute positional information
    generator = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(num_pairs):
        src = torch.randint(2, num_embeddings, (seq_len,), generator=generator)
        tgt = src.flip(0)
        pairs.append((src, tgt))
    return pairs


def make_shift_pairs(seq_len, num_pairs, num_embeddings, lag, seed):
    # tgt[i] = src[i-lag] (cyclic) -- a fixed relative offset regardless of
    # absolute position, i.e. exactly the pattern RoPE's relative encoding targets
    generator = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(num_pairs):
        src = torch.randint(2, num_embeddings, (seq_len,), generator=generator)
        tgt = torch.roll(src, shifts=lag, dims=0)
        pairs.append((src, tgt))
    return pairs


def train_model(module, config, pairs, device, seed):
    torch.manual_seed(seed)

    model = module.Transformer(
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        num_embeddings_input=config["num_embeddings"],
        num_embeddings_output=config["num_embeddings"],
        pad_token_id=config["pad_token_id"],
    )
    model.to(device)

    dataset = module.TranslationDataset(pairs, bos_token_id=config["bos_token_id"], pad_token_id=config["pad_token_id"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                             generator=torch.Generator().manual_seed(seed))

    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=partial(module.lr_lambda, d_model=config["d_model"], warmup_steps=config["warmup_steps"]))
    loss_fn = nn.CrossEntropyLoss(ignore_index=config["pad_token_id"], label_smoothing=0.1)

    history = []
    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        for src_batch, decoder_input_batch, loss_target_batch in dataloader:
            src_batch = src_batch.to(device)
            decoder_input_batch = decoder_input_batch.to(device)
            loss_target_batch = loss_target_batch.to(device)

            optimizer.zero_grad()
            predictions = model(src_batch, decoder_input_batch)
            loss = loss_fn(predictions.reshape(-1, predictions.shape[-1]), loss_target_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        history.append(epoch_loss / len(dataloader))

    return model, history


@torch.no_grad()
def token_accuracy(module, model, pairs, seq_len, config, device):
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    exact_matches = 0
    for src, tgt in pairs:
        prediction = module.predict(
            model, src, bos_token_id=config["bos_token_id"], pad_token_id=config["pad_token_id"],
            seq_len=seq_len, max_len=seq_len - 1, device=device,
        )
        target = tgt[:len(prediction)].tolist()
        matches = [p == t for p, t in zip(prediction, target)]
        correct_tokens += sum(matches)
        total_tokens += len(matches)
        exact_matches += int(all(matches))
    return correct_tokens / total_tokens, exact_matches / len(pairs)


def extend_context(module, model, new_seq_len, d_model, heads, device):
    # positional info is a closed-form function of position, so it can be regenerated
    # for lengths never seen during training without touching any learned weights
    if hasattr(model, "positional_encoding"):
        model.positional_encoding = module.Transformer._generate_positional_encoding(model, new_seq_len, d_model).to(device)
    if hasattr(model, "rope_sin"):
        rope_sin, rope_cos = module.Transformer.generate_rope_encoding(new_seq_len, d_model // heads)
        model.rope_sin = rope_sin.to(device)
        model.rope_cos = rope_cos.to(device)


def run_task(task_name, make_pairs, config, device):
    print(f"\n{'=' * 70}\ntask: {task_name}\n{'=' * 70}")

    seed = 0
    train_pairs = make_pairs(config["seq_len"], num_pairs=3000, num_embeddings=config["num_embeddings"], seed=seed)
    eval_pairs = make_pairs(config["seq_len"], num_pairs=300, num_embeddings=config["num_embeddings"], seed=seed + 1)

    results = {}
    for name, module in [("baseline (absolute sinusoidal PE)", baseline), ("roformer (RoPE)", roformer)]:
        model, history = train_model(module, config, train_pairs, device, seed)
        token_acc, exact_acc = token_accuracy(module, model, eval_pairs, config["seq_len"], config, device)
        results[name] = {"module": module, "model": model, "history": history}
        print(f"{name}: final loss {history[-1]:.4f}, "
              f"per-token accuracy {token_acc:.2%}, exact-match accuracy {exact_acc:.2%} (trained length)")

    print("\nloss every 5 epochs:")
    print(f"{'epoch':>6} " + " ".join(f"{name.split(' ')[0]:>10}" for name in results))
    for epoch in range(0, config["epochs"], 5):
        row = [f"{epoch + 1:>6}"]
        for name in results:
            row.append(f"{results[name]['history'][epoch]:>10.4f}")
        print(" ".join(row))

    # length extrapolation: evaluate at a sequence length the models never trained on,
    # regenerating each model's closed-form positional tables for the new length.
    # this is the property RoPE's relative-position formulation is specifically meant to help with.
    extrapolate_len = int(config["seq_len"] * 1.5)
    extrapolate_pairs = make_pairs(extrapolate_len, num_pairs=300, num_embeddings=config["num_embeddings"], seed=seed + 2)

    print(f"\nextrapolation to seq_len={extrapolate_len} (trained at seq_len={config['seq_len']}):")
    for name, result in results.items():
        extend_context(result["module"], result["model"], extrapolate_len, config["d_model"], config["heads"], device)
        token_acc, exact_acc = token_accuracy(result["module"], result["model"], extrapolate_pairs, extrapolate_len, config, device)
        print(f"{name}: per-token accuracy {token_acc:.2%}, exact-match accuracy {exact_acc:.2%} (extrapolated length)")


def main():
    config = {
        "d_model": 64,
        "seq_len": 32,
        "layers": 2,
        "heads": 4,
        "num_embeddings": 100,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "batch_size": 32,
        "epochs": 30,
        "warmup_steps": 400,
    }
    device = baseline.get_device()
    print(f"using device: {device}")

    tasks = [
        ("reverse-copy (absolute-position task)",
         lambda seq_len, num_pairs, num_embeddings, seed: make_reverse_pairs(seq_len, num_pairs, num_embeddings, seed)),
        ("shift-copy, lag=5 (relative-position task)",
         lambda seq_len, num_pairs, num_embeddings, seed: make_shift_pairs(seq_len, num_pairs, num_embeddings, lag=5, seed=seed)),
    ]

    for task_name, make_pairs in tasks:
        run_task(task_name, make_pairs, config, device)


if __name__ == "__main__":
    main()
