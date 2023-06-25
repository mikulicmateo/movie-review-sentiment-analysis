import torch.utils.data
from transformers import DistilBertTokenizer, AdamW

from BERT_Dataset_Class import BERTDataset
from Distil_BERT_Model_Class import DistilBERTModel
from data_util import load_raw_text_data
from trainer_util import *


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)

    reviews, labels = load_raw_text_data()

    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    token_lens = []

    # Iterate through the content slide
    for txt in reviews:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens.append(len(tokens))

    max_length = np.max(token_lens)

    full_dataset = BERTDataset(reviews, labels, tokenizer, max_length)
    training_data, validation_data, test_data = torch.utils.data.random_split(full_dataset, [40_000, 9_000, 1_000])
    batch_size = 16
    num_workers = 6

    training_loader = create_data_loader(training_data, batch_size=batch_size, num_workers=num_workers)
    validation_loader = create_data_loader(validation_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = create_data_loader(test_data, batch_size=batch_size, num_workers=num_workers)

    model = DistilBERTModel()
    model = model.to(device)

    EPOCHS = 7

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss().to(device)
    train(model, optimizer, training_loader, validation_loader, device, loss_fn, len(training_data),
          len(validation_data), EPOCHS)

    test_acc, test_loss = evaluate_epoch(
        model,
        test_loader,
        device,
        loss_fn,
        len(test_data)
    )
    print(f"Test accuracy: {test_acc}, Test loss {test_loss}")


if __name__ == "__main__":
    main()
