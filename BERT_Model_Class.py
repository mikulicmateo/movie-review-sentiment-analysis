from torch import nn
from transformers import BertModel


class BERTModel(nn.Module):

    # Constructor class
    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        obj = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #  Add a dropout layer
        output = self.drop(obj.pooler_output)
        return self.out(output)
