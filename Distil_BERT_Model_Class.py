from torch import nn
from transformers import DistilBertModel


class DistilBERTModel(nn.Module):

    # Constructor class
    def __init__(self):
        super(DistilBERTModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = nn.Dropout(p=0.7)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        distilbert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.out(output)
