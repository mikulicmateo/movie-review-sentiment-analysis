from torch import nn
from transformers import BertModel


class BERTModel(nn.Module):

    def __init__(self):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(p=0.7)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(output.pooler_output)
        return self.out(output)
