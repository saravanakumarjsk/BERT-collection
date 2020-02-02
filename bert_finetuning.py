
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# sentence = 'the boy likes to play'
# # Step 1: Tokenize
# tokens = tokenizer.tokenize(sentence)
# # Step 2: Add [CLS] and [SEP]
# tokens = ['[CLS]'] + tokens + ['[SEP]']
# # Step 3: Pad tokens
# padded_tokens = tokens + ['[PAD]' for _ in range(20 - len(tokens))]
# attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
# # Step 4: Segment ids
# seg_ids = [0 for _ in range(len(padded_tokens))] #Optional!
# # Step 5: Get BERT vocabulary index for each token
# token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)

# token_ids = torch.tensor(token_ids).unsqueeze(0)
# attn_mask = torch.tensor(attn_mask).unsqueeze(0)
# seg_ids = torch.tensor(seg_ids).unsqueeze(0)

# # Feed them to bert
# hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask,\
#                                   token_type_ids = seg_ids)
# print(hidden_reps.shape)
# print(cls_head.shape)



"""## Dataset Class and Data Loaders"""

# !pip install wget

import wget
import os

print('Downloading dataset...')

# The URL for the dataset zip file.
url = 'https://raw.githubusercontent.com/theneuralbeing/bert-finetuning-webinar/master/data.zip'

# Download the file and unzip it (if we haven't already)
if not os.path.exists('./data.zip'):
    wget.download(url, './data.zip')
    !unzip -q data.zip
    print('Unzipped Dataset')

from torch.utils.data import Dataset, DataLoader

class LoadDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter=',')

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Define the Maxlength for padding/truncating
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'review']
        label = self.df.loc[index, 'sentiment']

        # Tokenize the sentence
        tokens = self.tokenizer.tokenize(sentence)

        # Inserting the CLS and SEP token at the beginning and end of the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Padding/truncating the sentences to the maximum length
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        # Convert the sequence to ids with BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Converting the list to a pytorch tensor
        tokens_ids_tensor = torch.tensor(tokens_ids)

        # Obtaining the attention mask
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label

# Creating instances of training and validation set
train_set = LoadDataset(filename = 'data/train.csv', maxlen = 64)
val_set = LoadDataset(filename = 'data/validation.csv', maxlen = 64)

# Creating intsances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size = 32, num_workers = 5)
val_loader = DataLoader(val_set, batch_size = 32, num_workers = 5)

"""## Building the Model"""

from torch import nn

class SentimentClassifier(nn.Module):

    def __init__(self, freeze_bert = True):
        super(SentimentClassifier, self).__init__()

        # Instantiating the BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Defining layers like dropout and linear
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Getting contextualized representations from BERT Layer
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        # Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]
        # print('CLS shape: ',cls_rep.shape)

        # Feeding cls_rep to the classifier layer
        logits = self.classifier(cls_rep)
        # print('Logits shape: ',logits.shape)

        return logits

model = SentimentClassifier()

"""## Training"""

from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr = 2e-5)

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

print(device)

# Defining a function for calculating accuracy
def logits_accuracy(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    preds = (probs > 0.5).long()
    acc = (preds.squeeze() == labels).float().mean()
    return acc

# Defining an evaluation function for training
def evaluate(net, criterion, val_loader, device):

    losses, accuracies = 0, 0

    # Setting model to evaluation mode
    net.eval()

    count = 0
    for (seq, attn_masks, labels) in val_loader:
        count += 1

        # Move inputs and targets to device
        seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

        # Get logit predictions
        val_logits = net(seq, attn_masks)

        # Calculate loss
        val_loss = criterion(val_logits.squeeze(-1), labels.float())
        losses += val_loss.item()

        # Calculate validation accuracy
        accuracies += logits_accuracy(val_logits, labels)

    return losses / count, accuracies / count

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

# Saving our model
import os

if not os.path.isdir(save_path):
    os.mkdir(save_path)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'model.pth')

# Loading the checkpoints for resuming training
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

"""## Prediction"""

# predictor
inference_file = torch.load('model.pth')
predictor = SentimentClassifier()
predictor.load_state_dict(inference_file['model_state_dict'])

def preprocess(sentence, maxlen=64):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)

    # Inserting the CLS and SEP token at the beginning and end of the sentence
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # Padding/truncating the sentences to the maximum length
    if len(tokens) < maxlen:
        tokens = tokens + ['[PAD]' for _ in range(maxlen - len(tokens))]
    else:
        tokens = tokens[:maxlen-1] + ['[SEP]']

    # Convert the sequence to ids with BERT Vocabulary
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Converting the list to a pytorch tensor
    tokens_ids_tensor = torch.tensor(tokens_ids).unsqueeze(0)

    # Obtaining the attention mask
    attn_mask = (tokens_ids_tensor != 0).long()

    return tokens_ids_tensor, attn_mask

# Defining an evaluation function for training
def predict(net, iseq, masks):
    device = 'cpu'
    # Setting model to evaluation mode
    net.eval()

    # Move inputs and targets to device
    iseq, masks = iseq.to(device), masks.to(device)

    # Get logit predictions
    p_logit = net(iseq, masks)

    probs = torch.sigmoid(p_logit.unsqueeze(-1))
    preds = (probs > 0.5).long().squeeze(0)


    return preds, probs

test_tokens, test_attn = preprocess('the literally love this movie ever')

pred, probability = predict(predictor, test_tokens, test_attn)
print(pred, probability)

