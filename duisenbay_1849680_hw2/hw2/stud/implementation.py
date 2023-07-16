import json
import random

import numpy as np
from typing import List, Tuple, Dict

from transformers import AlbertTokenizerFast, AlbertForTokenClassification

import torch
import os
# from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm


from model import Model


# SEED = 1234

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.random.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


# device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_folder = './model/'
weights_path = os.path.join(model_folder, 'state_6.pt')


class HyperParams():

    language_model_name = "albert-base-v1"
    num_labels = 28
    recurrent_hidden_layer_size = 768
    mlp_hidden_size = 300
    predicate_emb_dim = 10
    fine_tune_lm = True

    batch_size = 8
    max_length = 32

hparams = HyperParams()


def build_model_34(language: str, device: str) -> Model:
    
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(language, hparams, device)


def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=False):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines




class ArgClassModule(torch.nn.Module):
    def __init__(self, hparams, *args, **kwargs) -> None:
        super().__init__()

        self.num_labels = hparams.num_labels
        # layers AutoModel
        self.bert_model = AlbertForTokenClassification.from_pretrained(hparams.language_model_name, output_hidden_states = True )
        # self.bert_model = AutoModel.from_pretrained(language_model_name, output_hidden_states = True )

        if not hparams.fine_tune_lm:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(0.2)

        self.predicate_embedder = torch.nn.Embedding(2, hparams.predicate_emb_dim)
        self.LSTM = torch.nn.LSTM(self.bert_model.config.hidden_size + hparams.predicate_emb_dim, hparams.recurrent_hidden_layer_size, batch_first=True, bidirectional=True)
        
        self.linear1 = torch.nn.Linear(2*hparams.recurrent_hidden_layer_size, hparams.mlp_hidden_size)
        self.linear2 = torch.nn.Linear(hparams.mlp_hidden_size, self.num_labels)
        self.sigmoid = torch.nn.Sigmoid()

        # self.classifier = torch.nn.Linear(
            # self.bert_model.config.hidden_size, hparams.num_labels, bias=False
        # )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        verb_indicator: torch.Tensor = None,
        compute_predictions: bool = False,
        compute_loss: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # group model inputs and pass to the model
        model_kwargs = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask
        }
        # not every model supports token_type_ids
        if token_type_ids != None :#or self.token_type_remove == True:
            model_kwargs["token_type_ids"] = token_type_ids

        # transformers_outputs = self.bert_model(**model_kwargs)
        transformers_outputs = self.bert_model(input_ids = input_ids, 
                                               token_type_ids = verb_indicator, 
                                               attention_mask = attention_mask,)
                                              #  output_all_encoded_layers=False)

        transformers_outputs_sum = transformers_outputs.hidden_states[-1]
        # transformers_outputs_sum = self.dropout(transformers_outputs_sum)

        predicate_embedding = self.predicate_embedder(verb_indicator)
        embedded_text_input = torch.cat([self.dropout(transformers_outputs_sum), predicate_embedding], 2)
        batch_size, sequence_length, _ = embedded_text_input.size()
        LSTM_output, (hidden_text, _) = self.LSTM(embedded_text_input)
        LSTM_output = LSTM_output.view(batch_size, sequence_length, 2, hidden_text.shape[-1])
        fwd_output = LSTM_output[:, :, 0, :]
        back_output = LSTM_output[:, :, 1, :]
        LSTM_combined = torch.cat((fwd_output, back_output), 2)
        lin1 = self.linear1(LSTM_combined)
        logits = self.linear2(self.sigmoid(lin1))
        # logits = self.classifier(transformers_outputs_sum)
        output = {"logits": logits}
        
        if compute_predictions:
            predictions = logits.argmax(dim=-1)
            output["predictions"] = predictions

        if compute_loss and labels is not None:
            output["loss"] = self.compute_loss(logits, labels)

        return output

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss of the model.
        Args:
            logits (`torch.Tensor`):
                The logits of the model.
            labels (`torch.Tensor`):
                The labels of the model.
        Returns:
            obj:`torch.Tensor`: The loss of the model.
        """
        return F.cross_entropy(
            logits.view(-1, self.num_labels),
            labels.view(-1),
            ignore_index=-100,
        )
class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str, hparams, device):
        # load the specific model for the input language
        self.language = language
        self.device = device
        self.tokenizer = AlbertTokenizerFast.from_pretrained(hparams.language_model_name, add_prefix_space=True)

        # load weights
        self.model = ArgClassModule(hparams) 
 

        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))
        # self.model.load_state_dict(torch.load(weights_path, map_location=device))
        
        # tags
        self.tags = ['[PAD]', '_', 'agent', 'asset', 'attribute', 'beneficiary', 'cause', 'co-agent', 'co-patient', 'co-theme', 'destination', 'experiencer', 'extent', 'goal', 'instrument', 'location', 'material', 'patient', 'product', 'purpose', 'recipient', 'result', 'source', 'stimulus', 'theme', 'time', 'topic','value']
        self.id2tag_dict = {i: key for i, key in enumerate(self.tags)}
        self.tag2id_dict = {key: value for value, key in self.id2tag_dict.items()}


    def tags2ids(self, labels: List) -> List[List]:
        return [self.tag2id_dict[label] for label in labels]
    def ids2tags(self, label_ids: List[List]) -> List[List]:
        return [[self.id2tag_dict[id] for id in ids] for ids in label_ids]
    

    def get_predicate_tokens(self, sentence: Dict ):
        
        predicates = list(sentence["predicates"])
        predicate_words = []
        predicate_pos = []
        for pos, pred in enumerate(predicates):
            if pred != '_':
                predicate_words.append(sentence['words'][pos])
                predicate_pos.append(pos)
        return predicate_words, predicate_pos

    def tokenize_funct(self, sentence, predicate_word):
        return self.tokenizer.encode_plus(text=sentence["words"],text_pair = [predicate_word], is_split_into_words=True, padding = 'max_length', truncation=True, max_length=hparams.max_length)

    def tokenize_and_align_labels(self, sentence, predicate_word, predicate_pos, label = None):
        tokenized_inputs = self.tokenize_funct(sentence, predicate_word)
        word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        pred_ind = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
                pred_ind.append(0)
            else:
                if word_idx == predicate_pos:
                    pred_ind.append(1)
                else:
                    pred_ind.append(0)
                if label!=None:
                    if word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
            previous_word_idx = word_idx
        if label!=None:
            tokenized_inputs["labels"] = label_ids
        tokenized_inputs["verb_indicator"] = pred_ind
        return tokenized_inputs

    def postprocess(self, prediction, word_ids):

        previous_word_idx = None
        label_ids = []

        for i, word_idx in enumerate(word_ids):  # Set the special tokens to -100.
            if not word_idx is None:
                if word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(prediction[i])
                    # label_ids.append(-100)
            previous_word_idx = word_idx
        # tokenized_inputs["labels"] = label_ids
        return label_ids

    def predict(self, sentence):

        # predictions = {}
        roles = {}
        self.model.eval()
        with torch.no_grad():
        # x = []
            predicate_words, predicate_pos = self.get_predicate_tokens(sentence)
            # in case of empty predicate
            if predicate_words==[]:
                predicate_words = ['[PAD]']
            for pred, pos in zip(predicate_words, predicate_pos):
                tokenized_inputs = self.tokenize_and_align_labels(sentence, pred, pos)
                # x.append(tokenized_inputs)
                batch = {k: torch.unsqueeze(torch.tensor(v), 0).to(self.device) for k, v in tokenized_inputs.items()}
          
                Y_hat = self.model(**batch, compute_predictions = True)
                output = torch.squeeze(Y_hat['predictions'],0)
                tags_per_sent =  self.ids2tags([[ (tag_dist).item() for tag_dist in output]])[0]

                roles[pos]=self.postprocess(tags_per_sent, tokenized_inputs.word_ids())


        # predictions[sentence_id] = {"roles": roles}
        return {"roles": roles}

        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence.
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                    }
        """
        # pass


