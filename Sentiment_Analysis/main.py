import csv
import os.path

import bitsandbytes.optim
import numpy as np
import peft
from bitsandbytes.optim import AdamW8bit
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, QuantoConfig, AutoModelForSequenceClassification, LlamaConfig, \
    AutoConfig, BitsAndBytesConfig, BatchEncoding, DistilBertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, SequenceClassifierOutput

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 64
PRE_BATCH_SIZE = 32
EMBEDDINGS_IN = "embeddings_llama3_in.npy"
EMBEDDINGS_OUT = "embeddings_llama3_out.npy"
EMBEDDINGS_TEST = "embeddings_llama3_test.npy"
CHECKPOINT = "llama3_embed_300E_0.074L.pt"


class RegressionClassifierForFinetune(nn.Module):
    def __init__(self, ptm: nn.Module, in_features: int, pad_token_id: int):
        super().__init__()
        self.ptm = ptm
        self.ptm.requires_grad_(False)
        self.classifier = nn.Sequential(nn.Softmax(1),
                                        nn.Linear(in_features, 100),
                                        nn.ReLU(),
                                        nn.Linear(100, 50),
                                        nn.ReLU(),
                                        nn.Linear(50, 25),
                                        nn.Sigmoid(),
                                        nn.Linear(25, 1)).to(dtype=ptm.dtype)
        self.pad_token_id = pad_token_id
        self.sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        res = self.ptm(input_ids=input_ids, attention_mask=kwargs["attention_mask"])
        last_hidden_state = res[0]  # (batch_size, seq_len, dim)

        # do classification on last non-padding token, see LlamaForSequenceClassification
        batch_size = input_ids.shape[0]
        sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

        pooled_output = self.classifier(pooled_output)
        # clamped = self.sigmoid(pooled_output) * 10
        loss_fct = nn.MSELoss()
        loss = loss_fct(pooled_output.squeeze(), kwargs["labels"])
        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_output,
            hidden_states=res.hidden_states,
            attentions=res.attentions,
        )


class RegressionClassifier(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_features, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 1))

    def forward(self, data):
        return self.classifier(data)


class ReviewTrainDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer):
        self.items = []
        with open(path, encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            self.max_tokens = 0
            for item in reader:
                title, body, rating = item
                tokens = tokenizer.encode_plus(f"{title}. {body}")
                self.items.append((tokens, float(rating)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class ReviewTestDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer):
        self.items = []
        with open(path, encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            self.max_tokens = 0
            for item in reader:
                title, body = item
                tokens = tokenizer.encode_plus(f"{title}. {body}")
                self.items.append((tokens, 0.0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main():
    torch.manual_seed(69420)
    train = True
    if train:
        if not os.path.isfile(EMBEDDINGS_IN) or not os.path.isfile(EMBEDDINGS_OUT):
            ptm, tokenizer = load_llama3()
            data_loader = load_training_data("train.csv", tokenizer)
            if not os.path.isfile(EMBEDDINGS_IN):
                generate_embeddings(EMBEDDINGS_IN, data_loader, ptm, tokenizer, 4096)

            if not os.path.isfile(EMBEDDINGS_OUT):
                cache_results(data_loader)

        model = RegressionClassifier(4096)
        dataset = TensorDataset(load_embeddings(EMBEDDINGS_IN), load_embeddings(EMBEDDINGS_OUT))
        training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
        data_loader = DataLoader(dataset=training_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=0, shuffle=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=0)
        train_model(data_loader, validation_loader, model)

    else:
        checkpoint = torch.load(CHECKPOINT)
        model = RegressionClassifier(4096)
        model.load_state_dict(checkpoint["model_state_dict"])

    if not os.path.isfile(EMBEDDINGS_TEST):
        ptm, tokenizer = load_llama3()
        data_loader = load_test_data("test_no_score.csv", tokenizer)
        generate_embeddings(EMBEDDINGS_TEST, data_loader, ptm, tokenizer, 4096)

    embeddings = load_embeddings(EMBEDDINGS_TEST)
    embeddings_dataset = TensorDataset(embeddings)
    data_loader = DataLoader(dataset=embeddings_dataset, batch_size=256, num_workers=0)
    run_inference(data_loader, model)


def cache_results(data_loader):
    results = np.zeros(len(data_loader.dataset))
    for (i, batch) in enumerate(data_loader):
        _, output_data = batch

        num_items = len(output_data)
        offset = i * data_loader.batch_size
        results[offset:offset + num_items] = output_data.squeeze().cpu().numpy()
    np.save(EMBEDDINGS_OUT, results)


def collate_fn(tokenizer, item):
    tokens, results = zip(*item)
    return (tokenizer.pad(tokens, padding="longest", return_attention_mask=True, return_tensors="pt"),
            torch.tensor(results))


def load_training_data(path, tokenizer) -> DataLoader:
    dataset = ReviewTrainDataset(path, tokenizer)
    return DataLoader(dataset=dataset, batch_size=PRE_BATCH_SIZE, num_workers=0,
                      collate_fn=lambda items: collate_fn(tokenizer, items))


def load_test_data(path, tokenizer) -> DataLoader:
    dataset = ReviewTestDataset(path, tokenizer)
    return DataLoader(dataset=dataset, batch_size=PRE_BATCH_SIZE, num_workers=0,
                      collate_fn=lambda items: collate_fn(tokenizer, items))


def load_distilbert() -> (nn.Module, AutoTokenizer):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased")
    config.num_labels = 1
    model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased",
                                      torch_dtype=torch.bfloat16,
                                      config=config
                                      # attn_implementation="flash_attention_2"
                                      )
    return model, tokenizer


def load_llama3() -> (nn.Module, AutoTokenizer):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token="insert_secret_token_here") #this is left out for security
    tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})

    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B", token="insert_secret_token_here")
    config.num_labels = 1
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                      device_map=DEVICE,
                                      quantization_config=double_quant_config,
                                      config=config,
                                      token="insert_secret_token_here")
    # freeze all except classification head
    # model.model.requires_grad_(False)

    # loftq_config = LoftQConfig(loftq_bits=4)
    # lora_config = LoraConfig(lora_alpha=8,
    #                          lora_dropout=0.05,
    #                          r=8,
    #                          bias="none",
    #                          # target_modules="all-linear",
    #                          task_type="SEQ_CLS",
    #                          # loftq_config=loftq_config,
    #                          # init_lora_weights="loftq",
    #                          inference_mode=False)
    # model = peft.get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    return model, tokenizer


def load_embeddings(path: str) -> torch.Tensor:
    embeddings = np.load(path)
    return torch.from_numpy(embeddings)


def generate_embeddings(path: str, data_loader: DataLoader, model: nn.Module, tokenizer: AutoTokenizer,
                        out_features: int):
    total_batches = len(data_loader)
    total_items = len(data_loader.dataset)
    embeddings = np.zeros((total_items, out_features))
    print(f"Generating embeddings of size {embeddings.shape}")
    model.eval()

    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(data_loader, total=total_batches)):
            input_data, output_data = batch
            input_data = input_data.to(DEVICE)
            input_ids = input_data.input_ids

            pred = model(input_ids=input_ids,
                         attention_mask=input_data.attention_mask)
            last_hidden_state = pred[0]  # (batch_size, seq_len, dim)

            # do classification on last non-padding token, see LlamaForSequenceClassification
            batch_size = input_ids.shape[0]
            sequence_lengths = torch.eq(input_ids, tokenizer.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(last_hidden_state.device)
            pooled_output = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

            num_items = pooled_output.shape[0]
            offset = i * data_loader.batch_size
            embeddings[offset:offset + num_items] = pooled_output.cpu().numpy()

    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")


def finetune_model(data_loader: DataLoader, model: nn.Module):
    model = model.to(DEVICE)
    model.train()
    n_epochs = 100
    learning_rate = 0.01
    loss_fn = nn.MSELoss()
    optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100)

    total_batches = len(data_loader)
    training_losses = np.zeros(total_batches)
    for epoch in tqdm(range(n_epochs), total=n_epochs):
        loss = None
        for (i, batch) in enumerate(tqdm(data_loader, total=total_batches)):
            input_data, output_data = batch
            input_data = input_data.to(DEVICE)
            output_data = output_data.to(device=DEVICE, dtype=torch.half)

            pred = model(input_ids=input_data.input_ids,
                         attention_mask=input_data.attention_mask,
                         labels=output_data
                         )
            loss = pred.loss
            training_losses[i] = loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            if i % 1 == 0:
                print(
                    f"Epoch: {epoch}/{n_epochs} pred: {pred.logits} loss: {loss}  [{i:>5d}/{total_batches:>5d}]")

        print(f"Epoch: {epoch}/{n_epochs} average training loss: {np.mean(training_losses):>7f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "checkpoint.pt")
        print("saved checkpoint")


def init_weights(layer: nn.Module):
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight)


def run_inference(data_loader: DataLoader, model: nn.Module):
    model.to(device=DEVICE, dtype=torch.bfloat16)
    model.eval()

    total_batches = len(data_loader)
    total_items = len(data_loader.dataset)
    results = np.empty(total_items)
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(data_loader, total=total_batches)):
            batch = batch[0].to(device=DEVICE, dtype=torch.bfloat16)
            res = model(batch).squeeze()
            res = torch.clamp(res, min=0.0, max=10.0)

            num_items = len(batch)
            offset = i * data_loader.batch_size
            results[offset:offset + num_items] = res.to(device="cpu", dtype=torch.float32).numpy()

    np.savetxt("results.txt", results, fmt='%f')


def train_model(data_loader: DataLoader, validation_loader: DataLoader, model: nn.Module):
    model = model.to(device=DEVICE, dtype=torch.bfloat16)
    # model.apply(init_weights)

    n_epochs = 30
    learning_rate = 0.001
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100)

    for epoch in tqdm(range(n_epochs), total=n_epochs):
        total_batches = len(data_loader)
        training_losses = np.zeros(total_batches)
        model.train()
        for (i, batch) in enumerate(tqdm(data_loader, total=total_batches)):
            input_data, output_data = batch
            input_data = input_data.to(device=DEVICE, dtype=torch.bfloat16)
            output_data = output_data.to(device=DEVICE, dtype=torch.bfloat16)

            pred = model(input_data)
            loss = loss_fn(pred.squeeze(), output_data.squeeze())
            training_losses[i] = loss.to(torch.float32).item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if i > 0 and i % 200 == 0:
            #     print(
            #        f"Epoch: {epoch}/{n_epochs} pred: {pred} loss: {training_losses[i]} [{i:>5d}/{total_batches:>5d}]")

        scheduler.step()

        model.eval()
        validation_batches = len(validation_loader)
        validation_losses = np.zeros(validation_batches)
        with torch.no_grad():
            for (i, batch) in enumerate(tqdm(validation_loader, total=validation_batches)):
                input_data, output_data = batch
                input_data = input_data.to(device=DEVICE, dtype=torch.bfloat16)
                output_data = output_data.to(device=DEVICE, dtype=torch.bfloat16)

                pred = model(input_data)
                loss = loss_fn(pred.squeeze(), output_data.squeeze())
                validation_losses[i] = loss.to(torch.float32).item()

        print(
            f"Epoch: {epoch}/{n_epochs} training loss: {np.mean(training_losses)} validation loss {np.mean(validation_losses)}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "checkpoint.pt")
        print("saved checkpoint")


if __name__ == "__main__":
    main()
