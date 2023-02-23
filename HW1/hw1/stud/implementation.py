import numpy as np
from typing import List, Tuple, Dict, Optional

from model import Model

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(42)

import os
import pandas as pd

from seqeval.metrics import accuracy_score, f1_score
import csv

#fix path to the model folder
model_folder = './model/'
embedding_path = os.path.join(model_folder, 'glove.6B.300d.txt')
weights_path = os.path.join(model_folder, 'state_3.pt')


class HyperParams():
	embedding_dim = 300
	hidden_size = 512
	dropout = 0.4
	batch_size = 32
	num_layers=1
	bi_dir = True
	stacked = True

hparams = HyperParams()


word_vector = pd.read_csv(embedding_path, sep=" ", header=None, engine='c', quoting=csv.QUOTE_NONE, index_col=0)
glove = {key: val.values for key, val in word_vector.T.items()}


def build_model(device: str) -> Model:
	model = NER(glove, hparams, device)
	model.to(device)
	student_model = StudentModel(model, glove, hparams, device)

	return student_model
	# STUDENT: return StudentModel()
	# STUDENT: your model MUST be loaded on the device "device" indicates
	# return RandomBaseline()

class NER(nn.Module):
	def __init__(self, glove: Dict, hparams, device):
		super(NER, self).__init__()
		self.device = device
		weight_matrix = self.get_weight_matrix(glove)
		self.embedding = nn.Embedding.from_pretrained(weight_matrix)
		self.embedding.requires_grad = False
		self.stacked = hparams.stacked

		_, embedding_dim = weight_matrix.shape

		num_dirs = 1

		if hparams.bi_dir:
			num_dirs = 2
		
		self.lstm = nn.LSTM(
			input_size=embedding_dim,
			hidden_size=hparams.hidden_size,
			num_layers=hparams.num_layers,
			batch_first=True,
			bidirectional = hparams.bi_dir
		)

		if hparams.stacked:
			self.lstm2 = nn.LSTM(
				input_size=hparams.hidden_size*num_dirs,
				hidden_size=hparams.hidden_size,
				num_layers=hparams.num_layers,
				batch_first=True,
				bidirectional = hparams.bi_dir
			)

		self.dropout = nn.Dropout(hparams.dropout)
		self.hidden_to_tag = nn.Linear(hparams.hidden_size*num_dirs , 14)
	def init_hidden(self, size):
		num_dirs = 1
		if hparams.bi_dir:
			num_dirs = 2
		hidden_a = Variable(torch.randn(hparams.num_layers*num_dirs, size, hparams.hidden_size))
		hidden_b = Variable(torch.randn(hparams.num_layers*num_dirs, size, hparams.hidden_size))
		return (hidden_a, hidden_b)

	def get_weight_matrix(self, glove:Dict) -> torch.Tensor:


		emb_dim = hparams.embedding_dim
		weight_matrix = np.asarray(list(glove.values()), dtype=np.float32)
		vocab_size, emb_dim  = weight_matrix.shape

		# for <UNK> take avg of vectors
		average_vec = np.mean(weight_matrix, axis=0)
		weight_matrix = np.concatenate((average_vec.reshape(1, emb_dim), weight_matrix), axis = 0)
		weight_matrix = np.concatenate((np.zeros((1, emb_dim)), weight_matrix), axis = 0)
		
		weights_matrix_torch = torch.tensor(weight_matrix, dtype=torch.float)
		return weights_matrix_torch

	def forward(self, X: torch.Tensor, X_lengths: torch.Tensor):
		batch_size, seq_len = X.shape
		(h_0, c_0) = self.init_hidden(X.shape[0])
		(h_2, c_2) = self.init_hidden(X.shape[0])

		embedded = self.embedding(X)
		X_lengths = torch.ones(batch_size, dtype=torch.int32)*X_lengths
		packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, X_lengths, batch_first=True)

		packed_out, (h_n, c_n)  = self.lstm(packed_embedded,  (h_0, c_0))

		if self.stacked:
			packed_out, (h_n, c_n)  = self.lstm2(packed_out,  (h_2, c_2))

		X, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

		
		X = X.contiguous()
		X = X.view(-1, X.shape[2])
		X = self.dropout(X)
		X = self.hidden_to_tag(X)

		X = F.log_softmax(X, dim=1)

		X = X.view(batch_size, seq_len, 14)
		Y_hat = X

		return Y_hat

	def neglogloss(self, Y_hat, Y):
		loss_f = torch.nn.NLLLoss()
		Y_hat = Y_hat.view(-1, Y_hat.shape[2])
		Y = Y.view(-1)
		loss = loss_f(Y_hat, Y)
		mask = (Y > 0).float()
		loss = loss * mask
		return torch.sum(loss) / torch.sum(mask)

 
		
class RandomBaseline(Model):
	options = [
		(3111, "B-CORP"),
		(3752, "B-CW"),
		(3571, "B-GRP"),
		(4799, "B-LOC"),
		(5397, "B-PER"),
		(2923, "B-PROD"),
		(3111, "I-CORP"),
		(6030, "I-CW"),
		(6467, "I-GRP"),
		(2751, "I-LOC"),
		(6141, "I-PER"),
		(1800, "I-PROD"),
		(203394, "O")
	]

	def __init__(self):
		self._options = [option[1] for option in self.options]
		self._weights = np.array([option[0] for option in self.options])
		self._weights = self._weights / self._weights.sum()

	def predict(self, tokens: List[List[str]]) -> List[List[str]]:
		return [
			[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
			for x in tokens
		]
  
class StudentModel(Model):
	# STUDENT: construct here your model
	# this class should be loading your weights and vocabulary
	def __init__(self, model, glove, hparam, device, batch_size=64):
		self.device = device
		self.model = model

		self.tags = ["<PAD>","B-PER", "B-LOC", "B-GRP", "B-CORP", "B-PROD", "B-CW", "I-PER", "I-LOC", "I-GRP", "I-CORP", "I-PROD", "I-CW", "O"]
        self.tag2id_dict = {key: value for value, key in enumerate(self.tags)}
        self.id2tag_dict = {value: key for value, key in enumerate(self.tags)}

        self.word2id_dict, self.id2word_dict = self.word2id_builder(list(glove.keys()))
        
		# load weights
		self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
		

	def tags2ids(self, labels: List[List]) -> List[List]:
		return [ [self.tag2id_dict[label] for label in label_sent] for label_sent in labels]

	def ids2tags(self, label_ids: List[List]) -> List[List]:
		return [[self.id2tag_dict[lid] for lid in lids] for lids in label_ids]

	def get_weight_matrix(self, glove: Dict) -> torch.Tensor:
		weight_matrix = np.asarray(list(glove.values()), dtype=np.float32)
		vocab_size, emb_dim  = weight_matrix.shape

		# for <UNK> take avg of vectors
		average_vec = np.mean(weight_matrix, axis=0)
		weight_matrix = np.concatenate((average_vec.reshape(1, emb_dim), weight_matrix), axis = 0)
		# for <PAD> instert all zeros
		weight_matrix = np.concatenate((np.zeros((1, emb_dim)), weight_matrix), axis = 0)
		# vocab_size  = weight_matrix.shape
		
		weights_matrix_torch = torch.tensor(weight_matrix, dtype=torch.float)
		return weights_matrix_torch
	
	def word2id_builder(self, word_dataset:List):
		word2id_dict = {"<UNK>":0, "<PAD>":1}
		id2word_dict = {0: "<UNK>",1: "<PAD>"}
		for i, word in enumerate(word_dataset):
			if word not in word2id_dict:
				id = len(word2id_dict)
				word2id_dict[word ] = id
				id2word_dict[id] = word
		return word2id_dict, id2word_dict

	def sentence2vector(self, sentenceList: List) -> List:
		return [self.word2id_dict[w] if w in self.word2id_dict 
                    else self.word2id_dict["<UNK>"] for w in sentenceList ]

	def predict(self, tokens: List[List[str]]) -> List[List[str]]:
		predictions = []
		#convert all sentences into vectors
		tokens_vectors = [self.sentence2vector(sentence) for sentence in tokens ]

		weights_matrix_torch = self.get_weight_matrix(glove)

		for sentence in tokens_vectors: #for each sentence
			d_tensor = torch.unsqueeze(torch.as_tensor(sentence), 0)
			out = self.model(d_tensor, d_tensor.shape[1]).squeeze()
			tags_per_sent = [ torch.argmax(tag_dist).item() for tag_dist in out ]
			predictions.append(tags_per_sent) #store all tags per each sentences
		return self.ids2tags(predictions)
		# STUDENT: implement here your predict function
		# remember to respect the same order of tokens!