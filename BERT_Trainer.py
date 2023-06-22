import numpy as np
from data_util import load_raw_text_data
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from BERT_Dataset_Class import BERTDataset
from BERT_Model_Class import BERTModel
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm


def create_data_loader(bert_dataset, batch_size, num_workers):
    return DataLoader(
        bert_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )


def train_epoch(model, training_dataloader, device, loss_fn, optimizer):
    model.train()
    train_loss = []
    loop = tqdm(training_dataloader, leave=False)

    for batch, labels in loop:
        batch = batch.to(device)
        labels = labels.to(device)

        predicted_data = model(batch)

        loss = loss_fn(predicted_data, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def validate_epoch(self, dataloader_index):
    self.resnet.eval()
    with torch.no_grad():
        val_losses = []
        loop = tqdm(self.validation_dataloader[dataloader_index], leave=False)
        for image_batch, labels in loop:
            labels = labels.to(self.device)
            for windowed_batch in image_batch:
                windowed_batch = windowed_batch.to(self.device)
                predicted_data = self.resnet(windowed_batch)
                loss = self.loss_fn(predicted_data, labels.float())
                val_losses.append(loss.detach().cpu().numpy())
                loop.set_postfix(loss=loss.item())
        # Evaluate global loss
    return np.mean(val_losses)


def test_model(self):
    self.resnet.eval()
    with torch.no_grad():
        val_losses = []
        loop = tqdm(self.test_dataloader, leave=False)
        for image_batch, labels in loop:
            labels = labels.to(self.device)
            for windowed_batch in image_batch:
                windowed_batch = windowed_batch.to(self.device)
                predicted_data = self.resnet(windowed_batch)
                loss = self.loss_fn(predicted_data, labels.float())
                val_losses.append(loss.detach().cpu().numpy())
                loop.set_postfix(loss=loss.item())
        # Evaluate global loss
    return np.mean(val_losses)


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)

    reviews, labels = load_raw_text_data()

    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    max_length = np.max([len(line) for line in reviews])

    full_dataset = BERTDataset(reviews, labels, tokenizer, max_length)
    training_data, validation_data, test_data = torch.utils.data.random_split(full_dataset, [40_000, 9_000, 1_000])

    training_loader = create_data_loader(training_data, batch_size=32, num_workers=6)
    validation_loader = create_data_loader(validation_data, batch_size=32, num_workers=6)
    test_loader = create_data_loader(test_data, batch_size=32, num_workers=6)

    model = BERTModel()
    model = model.to(device)






if __name__ == "__main__":
    main()


