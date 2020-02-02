import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.nn as nn 
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class LoadDataset(Dataset):

    def __init__(self, filename, maxlen):
        self.df = pd.read_csv(filename, delimeter = ',')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, 'review']
        label = self.df.loc[index, 'sentiment']

        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        if len(tokens) < maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = token[:self.maxlen - 1] + ['[SEP]']
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokens_ids_tensor = torch.tensor(token_ids)

        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


train_set = LoadDataset(filename = 'data/train.csv', maxlen = 64)
train_set = LoadDataset(filename = 'data/validation.csv', maxlen = 64)

train_loader = DataLoader(train_set, batch_size = 32, num_workers = 4)
val_loader = DataLoader(val_set, batch_size = 32, num_workers = 4)

class SentimentClassification(nn.Module):

    def __init__(self, freeze_bert = True):
        super(SentimentClassification, self).__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)

    def forward(self, seq, attn_mask):
        cont_reps, _  = self.bert_layer(seq, attention_mask = attn_masks)
        cls_reps = cont_reps[:, 0]
        logits = self.classifier(cls_reps)

        return logits

model = SentimentClassification()

criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr = 2e-5)

device = 'cuda'


def predicted_accuracy(predict, labels):
    prob = torch.sigmoid(logits.unsqueeze(-1))
    pred = (prob > 0.5).long()
    acc = (pred.squeeze(-1) == labels).float().mean()
    return acc

def evaluate(net, criterion, val_loader, device):
    losses, accuracy = 0, 0

    net.eval()

    for (seq, attn_mask, label) in val_loader:

        seq, attn_mask, label = seq.to(device), attn_mask.to(device), label.to(device)

        predict = net(seq, attn_mask)

        # calculate loss
        loss = criterion(predict.squeeze(-1), labels.float())
        losses += loss.item()

        accuracy += predicted_accuracy(predict, labels)

    return losses / count, accuracy / count

from time import time

def train(net, criterion, optimizer, train_loader, val_loader, device, epochs=4, print_every=100):

    # Move model to device
    net.to(device)
    # Setting model to training mode
    net.train()

    print('========== ========== STARTING TRAINING ========== ==========')

    for epoch in range(epochs):

        print('\n\n========== EPOCH {} =========='.format(epoch))
        t1 = time()

        for i, (seq, attn_masks, labels) in enumerate(train_loader):

            # Clear gradients
            optimizer.zero_grad()

            # Moving tensors to device
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            # Obtaining the logits from the model
            logits = net(seq,attn_masks)

            # Calculating the loss
            loss = criterion(logits.squeeze(-1), labels.float())

            # Backpropagating the gradients
            loss.backward()

            # Clipping gradients to tackle exploding gradients
            nn.utils.clip_grad_norm_(net.parameters(), 1)

            # Optimization step
            optimizer.step()

            if (i + 1) % print_every == 0:
                print("Iteration {} ==== Loss: {}".format(i+1, loss.item()))

        t2 = time()
        print('Time Taken for Epoch: {}'.format(t2-t1))
        print('\n========== Validating ==========')
        mean_val_loss, mean_val_acc = evaluate(net, criterion, val_loader, device)
        print("Validation Loss: {}\nValidation Accuracy: {}".format(mean_val_loss, mean_val_acc))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # starting training
train(model, criterion, optimizer, train_loader, val_loader, device, epochs=1, print_every=100)

import os

save_path = os.getcwd()

if not os.path.isdir(save_path):
    os.mkdir(save_path)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
}, 'model.pth')

checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

inference = torch.load('model.pth')
predictor = SentimentClassification()
predictor.load_state_dict(inference['model_state_dict'])

def preprocess(sentence, maxlen = 64):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    if len(tokens) < maxlen:
        tokens = tokens + ['[PAD]' for _ in range(maxlen - len(tokens))]
    else:
        tokens = tokens[:maxlen]
    
    tokens_ids = tokenizer.convert_tokens_to_ids()
    attn_mask = (token_ids != 0).long()
    return token_ids, attn_mask

def predict(net, token_ids, masks):
    device = 'cpu'
    # Setting model to evaluation mode
    net.eval()

    # Move inputs and targets to device
    token_ids, masks = token_ids.to(device), masks.to(device)

    # Get logit predictions
    p_logit = net(token_ids, masks)

    probs = torch.sigmoid(p_logit.unsqueeze(-1))
    preds = (probs > 0.5).long().squeeze(0)


    return preds, probs

test_tokens, test_attn = preprocess('the literally love this movie ever')

pred, probability = predict(predictor, test_tokens, test_attn)
print(pred, probability)








