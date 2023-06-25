import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def create_data_loader(bert_dataset, batch_size, num_workers):
    return DataLoader(
        bert_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )


def load_model(model_path, model, optimizer):
    model_dict = torch.load(model_path)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("cuda not available!")
        return None, None, None

    model.load_state_dict(model_dict['model_state'])
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer_state'])

    return model, optimizer


def create_model_state_dict(epoch, train_loss, val_loss, model, optimizer):
    model_state = {
        'time': str(datetime.datetime.now()),
        'model_state': model.state_dict(),
        'model_name': type(model).__name__,
        'optimizer_state': optimizer.state_dict(),
        'optimizer_name': type(optimizer).__name__,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    return model_state


def save_model(train_loss, val_loss, epoch, best, model, optimizer):
    model_state = create_model_state_dict(epoch, train_loss, val_loss, model, optimizer)
    torch.save(model_state, "Transformers/Models/last-BERT.pt")
    if best:
        torch.save(model_state, "Transformers/Models/best-BERT.pt")


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


def train(model, optimizer, training_dataloader, validation_dataloader, device, loss_fn, train_n_examples,
          val_n_examples, epochs_to_train=10, start_epoch=1, val_step=1):
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
        print(f"Training Loss: {train_loss}, Training accuracy: {train_acc}")

        # if epoch % val_step == 0:
        val_acc, val_loss = evaluate_epoch(model, validation_dataloader, device, loss_fn, val_n_examples)
        print(f"Validation Loss: {val_loss}, Val accuracy: {val_acc}")

        save_model(train_loss, val_loss, epoch, False, model, optimizer)
        if val_loss < best_val_loss:
            save_model(train_loss, val_loss, epoch, True, model, optimizer)
            best_val_loss = val_loss
            best_epoch = epoch

        # save_model(train_loss, val_loss, epoch, best=False)
        if val_acc > best_val_acc:
            # save_model(train_loss, val_loss, epoch, best=True)
            best_val_acc = val_acc
            best_epoch = epoch

        if epoch - best_epoch > val_step * 3:
            print(f"Early stopping at epoch {epoch}")
            early_stop = True
            break

        history['train_acc'].append(train_acc.cpu().detach().numpy())
        history['val_acc'].append(val_acc.cpu().detach().numpy())

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
    plt.show()

    print("Finished training")
