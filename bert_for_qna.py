import os
# !pip install ../input/sacremoses/sacremoses-master/
# !pip install ../input/transformers/transformers-master/

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import gc
import re
import os
import sys
import time
import pickle
import random
import unidecode
from tqdm import tqdm
tqdm.pandas()
from scipy.stats import spearmanr
from gensim.models import Word2Vec
from flashtext import KeywordProcessor
from keras.preprocessing import text, sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification, BertConfig,
    WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from math import floor, ceil

train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

sub = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

PUNCTS = {
            '》', '〞', '¢', '‹', '╦', '║', '♪', 'Ø', '╩', '\\', '★', '＋', 'ï', '<', '?', '％', '+', '„', 'α', '*', '〰', '｟', '¹', '●', '〗', ']', '▾', '■', '〙', '↓', '´', '【', 'ᴵ',
            '"', '）', '｀', '│', '¤', '²', '‡', '¿', '–', '」', '╔', '〾', '%', '¾', '←', '〔', '＿', '’', '-', ':', '‧', '｛', 'β', '（', '─', 'à', 'â', '､', '•', '；', '☆', '／', 'π',
            'é', '╗', '＾', '▪', ',', '►', '/', '〚', '¶', '♦', '™', '}', '″', '＂', '『', '▬', '±', '«', '“', '÷', '×', '^', '!', '╣', '▲', '・', '░', '′', '〝', '‛', '√', ';', '】', '▼',
            '.', '~', '`', '。', 'ə', '］', '，', '{', '～', '！', '†', '‘', '﹏', '═', '｣', '〕', '〜', '＼', '▒', '＄', '♥', '〛', '≤', '∞', '_', '[', '＆', '→', '»', '－', '＝', '§', '⋅',
            '▓', '&', 'Â', '＞', '〃', '|', '¦', '—', '╚', '〖', '―', '¸', '³', '®', '｠', '¨', '‟', '＊', '£', '#', 'Ã', "'", '▀', '·', '？', '、', '█', '”', '＃', '⊕', '=', '〟', '½', '』',
            '［', '$', ')', 'θ', '@', '›', '＠', '｝', '¬', '…', '¼', '：', '¥', '❤', '€', '−', '＜', '(', '〘', '▄', '＇', '>', '₤', '₹', '∅', 'è', '〿', '「', '©', '｢', '∙', '°', '｜', '¡',
            '↑', 'º', '¯', '♫', '#'
          }


mispell_dict = {"aren't" : "are not", "can't" : "cannot", "couldn't" : "could not",
"couldnt" : "could not", "didn't" : "did not", "doesn't" : "does not",
"doesnt" : "does not", "don't" : "do not", "hadn't" : "had not", "hasn't" : "has not",
"haven't" : "have not", "havent" : "have not", "he'd" : "he would", "he'll" : "he will", "he's" : "he is", "i'd" : "I would",
"i'd" : "I had", "i'll" : "I will", "i'm" : "I am", "isn't" : "is not", "it's" : "it is",
"it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not", "mustn't" : "must not",
"shan't" : "shall not", "she'd" : "she would", "she'll" : "she will", "she's" : "she is", "shouldn't" : "should not", "shouldnt" : "should not",
"that's" : "that is", "thats" : "that is", "there's" : "there is", "theres" : "there is", "they'd" : "they would", "they'll" : "they will",
"they're" : "they are", "theyre":  "they are", "they've" : "they have", "we'd" : "we would", "we're" : "we are", "weren't" : "were not",
"we've" : "we have", "what'll" : "what will", "what're" : "what are", "what's" : "what is", "what've" : "what have", "where's" : "where is",
"who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is", "who've" : "who have", "won't" : "will not", "wouldn't" : "would not", "you'd" : "you would",
"you'll" : "you will", "you're" : "you are", "you've" : "you have", "'re": " are", "wasn't": "was not", "we'll":" will", "didn't": "did not", "tryin'":"trying"}


def clean_punct(text):
  text = str(text)
  for punct in PUNCTS:
    text = text.replace(punct, ' {} '.format(punct))

  return text

kp = KeywordProcessor(case_sensitive=True)

for k, v in mispell_dict.items():
    kp.add_keyword(k, v)

def preprocessing(text):
    text = text.lower()
    text = re.sub(r'(\&lt)|(\&gt)', ' ', text)

    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' url ', text)
    text = kp.replace_keywords(text)
    text = clean_punct(text)
    text = re.sub(r'\n\r', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    return text

train['clean_title'] = train['question_title'].apply(lambda x : preprocessing(x))
train['clean_body'] = train['question_body'].apply(lambda x : preprocessing(x))
train['clean_answer'] = train['answer'].apply(lambda x : preprocessing(x))

test['clean_title'] = test['question_title'].apply(lambda x : preprocessing(x))
test['clean_body'] = test['question_body'].apply(lambda x : preprocessing(x))
test['clean_answer'] = test['answer'].apply(lambda x : preprocessing(x))

y_columns = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']

tokenizer = BertTokenizer.from_pretrained("../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt")

def trim_input(tokenizer, title, body, answer, max_seq_length=512):
    all_title = []
    all_body = []
    all_answer = []
    for t, b, a in tqdm(zip(title, body, answer), total=len(title)):

        tokenizer_t = tokenizer.tokenize(t)
        tokenizer_b = tokenizer.tokenize(b)
        tokenizer_a = tokenizer.tokenize(a)

        t_len = len(tokenizer_t)
        b_len = len(tokenizer_b)
        a_len = len(tokenizer_a)

        t_max_len=TITLE_MAX_LEN
        b_max_len=BODY_MAX_LEN
        a_max_len=ANSWER_MAX_LEN

        if (t_len+b_len+a_len+4) > max_seq_length:

            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len)/2)
                b_max_len = b_max_len + ceil((t_max_len - t_len)/2)
            else:
                t_new_len = t_max_len

            if a_max_len > a_len:
                a_new_len = a_len
                b_new_len = b_max_len + (a_max_len - a_len)
            elif b_max_len > b_len:
                a_new_len = a_max_len + (b_max_len - b_len)
                b_new_len = b_len
            else:
                a_new_len = a_max_len
                b_new_len = b_max_len


            if t_new_len+a_new_len+b_new_len+4 != max_seq_length:
                raise ValueError("New sequence length should be %d, but is %d"%(max_seq_length, (t_new_len + a_new_len + b_new_len + 4)))



            tokenizer_t = tokenizer_t[:t_new_len]
            tokenizer_b = tokenizer_b[:b_new_len]
            tokenizer_a = tokenizer_a[:a_new_len]

        all_title.append(tokenizer_t)
        all_body.append(tokenizer_b)
        all_answer.append(tokenizer_a)

    return all_title, all_body, all_answer

def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


# def convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
#     """Converts tokenized input to ids, masks and segments for BERT"""

#     stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

#     input_ids = get_ids(stoken, tokenizer, max_sequence_length)
#     input_masks = get_masks(stoken, max_sequence_length)
#     input_segments = get_segments(stoken, max_sequence_length)

#     return [input_ids, input_masks, input_segments]

def convert_lines(tokenizer, title, body, answer, max_seq_length=512):
    title, body, answer = trim_input(tokenizer, title, body, answer)
#     max_seq_length -= 4
    all_tokens = []
    longer = 0

    all_tokens = []
    input_ids, input_masks, input_segments = [], [], []
    for i, (t, b, a) in tqdm(enumerate(zip(title, body, answer)), total=len(title)):
        stoken = ["[CLS]"] + t + ["[SEP]"] + b + ["[SEP]"] + a + ["[SEP]"]

        ids = get_ids(stoken, tokenizer, max_seq_length)
        masks = get_masks(stoken, max_seq_length)
        segments = get_segments(stoken, max_seq_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]

TITLE_MAX_LEN = 30
BODY_MAX_LEN = 239
ANSWER_MAX_LEN = 239
# title, body, answer = trim_input(tokenizer, train['clean_title'], train['clean_body'], train['clean_answer'])
# title, body, answer = trim_input(tokenizer, test['clean_title'], test['clean_body'], test['clean_answer'])

x_train = convert_lines(tokenizer, train['clean_title'], train['clean_body'], train['clean_answer'])
x_test = convert_lines(tokenizer, test['clean_title'], test['clean_body'], test['clean_answer'])

# x_train_body = convert_lines(tokenizer, train['clean_body'], BODY_MAX_LEN)
# x_train_answer = convert_lines(tokenizer, train['clean_answer'], ANSWER_MAX_LEN)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class TextDataset(torch.utils.data.TensorDataset):

    def __init__(self, x_train, idxs, targets=None):
        self.input_ids = x_train[0][idxs]
        self.input_masks = x_train[1][idxs]
        self.input_segments = x_train[2][idxs]
        self.targets = targets[idxs] if targets is not None else np.zeros((x_train[0].shape[0], 30))

    def __getitem__(self, idx):
        input_ids =  self.input_ids[idx]
        input_masks = self.input_masks[idx]
        input_segments = self.input_segments[idx]

        target = self.targets[idx]

        return input_ids, input_masks, input_segments, target

    def __len__(self):
        return len(self.input_ids)

accumulation_steps = 1

SEED = 2020
NFOLDS = 3
BATCH_SIZE = 6
EPOCHS = 4
LR = 3e-5
seed_everything(SEED)

test_loader = torch.utils.data.DataLoader(TextDataset(x_test, test.index),
                          batch_size=BATCH_SIZE, shuffle=False)

gc.collect()

bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'
bert_config = BertConfig.from_json_file(bert_model_config)
bert_config.num_labels = 30

y = train.loc[:, y_columns].values

oof = np.zeros((len(train), 30))
test_pred = np.zeros((len(test), 30))


kf = list(KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED).split(train))
k = 0
for i, (train_idx, valid_idx) in enumerate(kf):
    print(f'fold {i+1}')
    gc.collect()

    train_loader = torch.utils.data.DataLoader(TextDataset(x_train, train_idx, y),
                          batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(TextDataset(x_train, valid_idx, y),
                          batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    net = BertForSequenceClassification.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/', config=bert_config)

    net.cuda()


    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(net.parameters(), lr=LR, eps=4e-5)

    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss = 0.0
        net.train()
        for data in train_loader:

            # get the inputs
            input_ids, input_masks, input_segments, labels = data
            pred = net(input_ids = input_ids.long().cuda(),
                             labels = None,
                             attention_mask = input_masks.cuda(),
                             token_type_ids = input_segments.cuda(),
                            )[0]
            loss = loss_fn(pred, labels.cuda())
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)


            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item()

        avg_val_loss = 0.0
        net.eval()

        valid_preds = np.zeros((len(valid_idx), 30))
        true_label = np.zeros((len(valid_idx), 30))
        for j, data in enumerate(val_loader):

            # get the inputs
#             body, answer, title, category, host, labels = data
#             content, labels = data
            input_ids, input_masks, input_segments, labels = data

            ## forward + backward + optimize
            pred = net(input_ids = input_ids.long().cuda(),
                             labels = None,
                             attention_mask = input_masks.cuda(),
                             token_type_ids = input_segments.cuda(),
                            )[0]
            loss_val = loss_fn(pred, labels.cuda())
            avg_val_loss += loss_val.item()

            valid_preds[j * BATCH_SIZE:(j+1) * BATCH_SIZE] = torch.sigmoid(pred).cpu().detach().numpy()
            true_label[j * BATCH_SIZE:(j+1) * BATCH_SIZE]  = labels


        score = 0
        for i in range(30):
            score += np.nan_to_num(
                    spearmanr(true_label[:, i], valid_preds[:, i]).correlation / 30)
        oof[valid_idx] = valid_preds
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t spearman={:.2f} \t time={:.2f}s'.format(
            epoch + 1, EPOCHS, avg_loss / len(train_loader), avg_val_loss / len(val_loader), score, elapsed_time))

    test_pred_fold = np.zeros((len(test), 30))
    k += 0
    torch.save(net.state_dict(), "bert_pytorch_folds_{}.pt".format(k))

    with torch.no_grad():
        for q, data in enumerate(test_loader):
            input_ids, input_masks, input_segments, labels = data
            y_pred = net(input_ids = input_ids.long().cuda(),
                             labels = None,
                             attention_mask = input_masks.cuda(),
                             token_type_ids = input_segments.cuda(),
                            )[0]
            test_pred_fold[q * BATCH_SIZE:(q+1) * BATCH_SIZE] = torch.sigmoid(y_pred).cpu().detach().numpy()

    torch.cuda.empty_cache()
    gc.collect()
    test_pred += test_pred_fold/NFOLDS

oof_score = 0
for i in range(30):
    oof_score += np.nan_to_num(
            spearmanr(y[:, i], oof[:, i]).correlation / 30)

oof_score

torch.cuda.empty_cache()
gc.collect()

sub.loc[:, y_columns] = test_pred
sub.to_csv('submission.csv', index=False)

sub.head()

