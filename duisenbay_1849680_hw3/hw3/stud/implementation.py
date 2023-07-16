import re

import numpy as np
from typing import List, Tuple, Dict

from model import Model
import json
import random
import torch
import os
from collections import Counter

from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
# checkpoint = "bert-large-uncased"
# checkpoint = 'albert-base-v1'
checkpoint = 'roberta-base'





device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_folder = './model/'
# weights_path = os.path.join(model_folder, 'albert_best.pt')
weights_path = os.path.join(model_folder, 'best.pt')


class HyperParams():
    language_model_name = checkpoint
    num_labels = 3
    batch_size = 8

hparams = HyperParams()


def build_model_123(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2 and 3 of the Coreference resolution pipeline.
            1: Ambiguous pronoun identification.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(True, True)


def build_model_23(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2 and 3 of the Coreference resolution pipeline.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(False, True)


def build_model_3(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements step 3 of the Coreference resolution pipeline.
            3: Coreference resolution
    """
    # return RandomBaseline(False, False)
    return StudentModel(hparams, device)


class RandomBaseline(Model):
    def __init__(self, predict_pronoun: bool, predict_entities: bool):
        self.pronouns_weights = {
            "his": 904,
            "her": 773,
            "he": 610,
            "she": 555,
            "him": 157,
        }
        self.predict_pronoun = predict_pronoun
        self.pred_entities = predict_entities

    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        predictions = []
        for sent in sentences:
            text = sent["text"]
            toks = re.sub("[.,'`()]", " ", text).split(" ")
            if self.predict_pronoun:
                prons = [
                    tok.lower() for tok in toks if tok.lower() in self.pronouns_weights
                ]
                if prons:
                    pron = np.random.choice(prons, 1, self.pronouns_weights)[0]
                    pron_offset = text.lower().index(pron)
                    if self.pred_entities:
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks
                        )
                    else:
                        entities = [sent["entity_A"], sent["entity_B"]]
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks, entities
                        )
                    predictions.append(((pron, pron_offset), entity))
                else:
                    predictions.append(((), ()))
            else:
                pron = sent["pron"]
                pron_offset = sent["p_offset"]
                if self.pred_entities:
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks
                    )
                else:
                    entities = [
                        (sent["entity_A"], sent["offset_A"]),
                        (sent["entity_B"], sent["offset_B"]),
                    ]
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks, entities
                    )
                predictions.append(((pron, pron_offset), entity))
        return predictions

    def predict_entity(self, predictions, pron, pron_offset, text, toks, entities=None):
        entities = (
            entities if entities is not None else self.predict_entities(entities, toks)
        )
        entity_idx = np.random.choice([0, len(entities) - 1], 1)[0]
        return entities[entity_idx]

    def predict_entities(self, entities, toks):
        offset = 0
        entities = []
        for tok in toks:
            if tok != "" and tok[0].isupper():
                entities.append((tok, offset))
            offset += len(tok) + 1
        return entities


class ProBERT(torch.nn.Module):
    def __init__(self, hparams, *args, **kwargs) -> None:
        super().__init__()
        self.num_labels = hparams.num_labels
        # layers AutoModel
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(hparams.language_model_name, output_hidden_states = True )

        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(
            self.bert_model.config.hidden_size, hparams.num_labels, bias=False)
        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            pronoun_ind: torch.Tensor = None,
            labels: torch.Tensor = None,
            compute_predictions: bool = False,
            compute_loss: bool = True,
            *args,
            **kwargs,
        ) -> torch.Tensor:
        model_kwargs = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }
        if token_type_ids != None :
            model_kwargs["token_type_ids"] = token_type_ids


        transformers_outputs = self.bert_model(input_ids = input_ids, 
                                               token_type_ids = token_type_ids, 
                                               attention_mask = attention_mask,)
        
        transformers_outputs = self.dropout(transformers_outputs.hidden_states[-1])
        pronoun_representations = torch.tensor([])
        for pronoun in pronoun_ind:

          p_repr = transformers_outputs[:,pronoun,:]

          if pronoun_representations.shape[0] ==  0:
              pronoun_representations = p_repr
          else:
              torch.stack((pronoun_representations, p_repr))

        classifier_output = self.classifier(pronoun_representations)
        logits = self.softmax(classifier_output)
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
        return F.cross_entropy(
            logits.view(-1, self.num_labels),
            labels.view(-1),
            ignore_index=-100,
            )


class StudentModel(Model):
    def __init__(self, hparams, device):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.language_model_name)

        self.model = ProBERT(hparams) 
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device(self.device)))


    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def get_label(self, example):
        if example['is_coref_A'] == 'TRUE':
            return 0 # entity A
        elif example['is_coref_B'] == 'TRUE':
            return 1 #entity B
        else:
            return 2 #Neither
    def tokenize_and_align_pronoun_pos(self, sentence, pronoun_id):
        tokenized_inputs = self.tokenizer(sentence)
        word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.

        new_pronoun_id = 0
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                new_pronoun_id += 1
            else:
                if word_idx == pronoun_id:
                    break
                else:
                    new_pronoun_id += 1

        tokenized_inputs['pronoun_ind'] = new_pronoun_id
        return tokenized_inputs

    def add_tags(self, sample):
        # <A> <B> <P> enclosing with tags
        out = ""
        pos_dict = {}

        pos_dict[sample['p_offset']] = ['<P>', len(sample['pron'])]
        pos_dict[sample['offset_A']] = ['<A>', len(sample['entity_A'])]
        pos_dict[sample['offset_B']] = ['<B>', len(sample['entity_B'])]
        positions = list(pos_dict.keys())
        positions.sort()

        start = 0
        end = 0
        pronoun_id = None

        for i, offset in enumerate(positions):
            end = offset
            [tag, word_len] = pos_dict[offset]
            out += sample['text'][start:end] + tag
            if tag == '<P>':
                pronoun_id = len(list(out.split(' ')))
                # pronoun_id = len(word_tokenize(out))
            out += " " + sample['text'][end:end+word_len] + " " +tag
            start = offset + word_len
            if i == 2:
                out +=  sample['text'][start:]
        return pronoun_id, out


    def postprocess(self, sentences, outputs):
        predictions = []
        for sent, pred in zip(sentences,outputs ):
            prediction = []
            prediction.append([sent['pron'], sent['p_offset']])
            if pred == 0: # entity A
                prediction.append([sent['entity_A'], sent['offset_A']])
            elif pred == 1: # entity B
                prediction.append([sent['entity_B'], sent['offset_B']])
            else:
                prediction.append([None, None])
            predictions.append(prediction)
        return predictions

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        # pass
        sentences = tokens
        self.model.eval()
        with torch.no_grad():
            sentences_pbar = tqdm(enumerate(sentences), total=len(sentences))
            out = []
            for i, sample in sentences_pbar:
                # print()
                # print("sentence: {}".format(sample))
                pronoun_id, sentence = self.add_tags(sample)
                input_s = self.tokenize_and_align_pronoun_pos(sentence, pronoun_id)
                # input['label'] = self.get_label(sample)
                batch = {k: torch.unsqueeze(torch.tensor(v), 0).to(device) for k, v in input_s.items()}
                Y_hat = self.model.forward(**batch, compute_predictions = True)
                output = torch.squeeze(Y_hat['predictions'],0)
                out.append(output)
            # print("len of out is ", len(out))
        predictions = self.postprocess(sentences, out)
        # print("len of predictions ", len(predictions))
            
        return predictions

