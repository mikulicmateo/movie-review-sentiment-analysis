import numpy as np
from data_util import load_raw_text_data
from transformers import BertTokenizer, AdamW
from torch import nn
from BERT_Dataset_Class import BERTDataset
from BERT_Model_Class import BERTModel
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt



def create_data_loader(bert_dataset, batch_size, num_workers):
    return DataLoader(
        bert_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )


def train_epoch(model, training_dataloader, device, loss_fn, optimizer, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    loop = tqdm(training_dataloader, leave=False)

    for d in loop:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()

        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def evaluate_epoch(model, validation_dataloader, device, loss_fn, n_examples):
    model.eval()
    with torch.no_grad():
        losses = []
        correct_predictions = 0
        loop = tqdm(validation_dataloader, leave=False)
        for d in loop:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Get model ouptuts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

def train(model, optimizer, training_dataloader, validation_dataloader, device, loss_fn, train_n_examples, val_n_examples, epochs_to_train=10, start_epoch=1, val_step=1):

    if model is None or optimizer is None:
        print("Model and Optimizer not initialized")
        return

    print('Going training!')
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    early_stop = False
    history = defaultdict(list)


    for epoch in range(start_epoch, start_epoch + epochs_to_train):

        print(f"Epoch {epoch}:")
        train_acc, train_loss = train_epoch(model, training_dataloader, device, loss_fn, optimizer, train_n_examples)
        print(f"Training Loss: {train_loss}, Training accuraccy: {train_acc}")

        #if epoch % val_step == 0:
        val_acc, val_loss = evaluate_epoch(model, validation_dataloader, device, loss_fn, val_n_examples)
        print(f"Validation Loss: {val_loss}")


        # #save_model(train_loss, val_loss, epoch, best=False)
        # if val_loss < best_val_loss:
        #     #save_model(train_loss, val_loss, epoch, best=True)
        #     best_val_loss = val_loss
        #     best_epoch = epoch

        #save_model(train_loss, val_loss, epoch, best=False)
        if val_acc > best_val_acc:
            #save_model(train_loss, val_loss, epoch, best=True)
            best_val_acc = val_acc
            best_epoch = epoch

        if epoch - best_epoch > val_step * 3:
            print(f"Early stopping at epoch {epoch}")
            early_stop = True
            break

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        print("---------------------------")

        if early_stop:
            break

    # Plot training and validation accuracy
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    # Graph chars
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])

    print("Finished training")

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

    EPOCHS = 10

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-05, correct_bias=False)
    loss_fn = nn.CrossEntropyLoss().to(device)
    train(model, optimizer, training_loader, validation_loader, device, loss_fn, len(training_data), len(validation_data),EPOCHS)

    test_acc, test_loss = evaluate_epoch(
        model,
        test_loader,
        device,
        loss_fn,
        len(test_data)
    )
    print(f"Training accuraccy: {test_acc}, Test loss {test_loss}")


if __name__ == "__main__":
    main()


