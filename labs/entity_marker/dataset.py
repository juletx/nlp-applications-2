import json

import torch
from torch.utils.data import Dataset, DataLoader

from pprint import pprint

from itertools import product


def load_json(file_path):
    with open(file_path, 'rt') as f:
        data = [json.loads(line) for line in f]
    return data


class ACEDataset(Dataset):

    def __init__(self, from_preprocessed=True):
        super().__init__()

        self.from_preprocessed = from_preprocessed

    def preprocessed2output(self):
        """ TODO Finish
        """
        if self.from_preprocessed:
            new_instances = []
            for inst in self.instances:
                doc_id = inst['doc_id']
                sent_id = inst['sent_id']
                tokens = inst['tokens']
                entity2id = {ent['id']: i for i, ent in enumerate(inst['entity_mentions'])}
                entities = [
                    [ent['start'], ent['end'], ent['entity_type'], ent['mention_type'], 1.0]
                    for ent in inst['entity_mentions']
                ]
                triggers, roles = [], []
                for i, event in enumerate(inst['event_mentions']):
                    triggers.append(
                        [event['trigger']['start'], event['trigger']['end'], event['event_type'], 1.0]
                    )
                    for arg in event['arguments']:
                        roles.append(
                            [ i, entity2id[arg['entity_id']], arg['role'], 1.0 ]    
                        )
                relations = []  # OIER
                for i, relation in enumerate(inst['relation_mentions']):
                    relations.append([entity2id[relation['arguments'][0]['entity_id']],
                                     entity2id[relation['arguments'][1]['entity_id']],
                                     relation['relation_subtype']])

                new_instances.append({
                    'doc_id': doc_id,
                    'sent_id': sent_id,
                    'tokens': tokens,
                    'graph': {
                        'entities': entities,
                        'triggers': triggers,
                        'relations': relations,
                        'roles': roles
                    }
                })

            self.from_preprocessed = False
            self.instances = new_instances

    def __getitem__(self, idx):
        return self.train_instances[idx]

    def __len__(self):
        return len(self.train_instances)


class ACERelationClassificationDataset(ACEDataset):
    label2id = {
        "ART:User-Owner-Inventor-Manufacturer": 0,
        "GEN-AFF:Citizen-Resident-Religion-Ethnicity": 1,
        "GEN-AFF:Org-Location": 2,
        "ORG-AFF:Employment": 3,
        "ORG-AFF:Founder": 4,
        "ORG-AFF:Investor-Shareholder": 5,
        "ORG-AFF:Membership": 6,
        "ORG-AFF:Ownership": 7,
        "ORG-AFF:Sports-Affiliation": 8,
        "ORG-AFF:Student-Alum": 9,
        "PART-WHOLE:Artifact": 10,
        "PART-WHOLE:Geographical": 11,
        "PART-WHOLE:Subsidiary": 12,
        "PER-SOC:Business": 13,
        "PER-SOC:Family": 14,
        "PER-SOC:Lasting-Personal": 15,
        "PHYS:Located": 16,
        "PHYS:Near": 17,
    }

    id2label = {value: key for key, value in label2id.items()}

    def __init__(self, file_path, tokenizer, from_preprocessed=True,
                 window_length=80000, ignore_labels=False):
        super().__init__(from_preprocessed=from_preprocessed)

        self.instances = load_json(file_path)
        self.window_length = window_length

        if from_preprocessed:
            self.train_instances = self._preprocess_data(self.instances)
        else:
            self.train_instances = self._preprocess_output_data(self.instances)

        self.ignore_labels = ignore_labels

        self.tokenizer = tokenizer
        self.instance2id = {inst['sent_id']: i for i, inst in enumerate(self.instances)}

    def _preprocess_data(self, instances):
        new_instances = []
        lengths, distances = [], []
        # optionally add entity-type information (see ACEArgumentClassificationDataset)
        # for that we need to add special token
        e1s, e1s = "<e1s>", "<e1e>"
        e2s, e2s = "<e2s>", "<e2e>"
        for inst in instances:
            entities = {ent['id']: ent for ent in inst['entity_mentions']}
            for relation in inst['relation_mentions']:
                label = relation['relation_subtype']
                args = relation['arguments']
                if entities[args[0]['entity_id']]['start'] > entities[args[1]['entity_id']]['start']:
                    args = [args[1], args[0]]
                tokens = inst['tokens'][:entities[args[0]['entity_id']]['start']] + \
                         [e1s] + inst['tokens'][entities[args[0]['entity_id']]['start']:entities[args[0]['entity_id']]['end']] + [e1s] + \
                         inst['tokens'][entities[args[0]['entity_id']]['end']:entities[args[1]['entity_id']]['start']] + \
                         [e2s] + inst['tokens'][entities[args[1]['entity_id']]['start']: entities[args[1]['entity_id']]['end']] + [e2s] + \
                         inst['tokens'][entities[args[1]['entity_id']]['end']:]
                lengths.append(len(tokens))
                #distances.append(abs(args[0]['start'] - args[1]['start']))
                new_instances.append(
                    {
                        'doc_id': inst['doc_id'],
                        'sent_id': inst['sent_id'],
                        'label': label,
                        'tokens': tokens,
                        'arg1_span': (entities[args[0]['entity_id']]['start'], entities[args[0]['entity_id']]['end']),
                        'arg2_span': (entities[args[1]['entity_id']]['start'], entities[args[1]['entity_id']]['end'])
                    }
                )
        return new_instances

    def _preprocess_output_data(self, instances):
        ## TBD
        new_instances = []
        return new_instances

    def collate_fn(self, instances, window_length=None):
        if not window_length:
            return_dict = self.tokenizer(
                [inst['tokens'] for inst in instances],
                is_split_into_words=True, return_tensors='pt',
                padding=True, truncation=True
            )
        else:
            tokens = []
            for inst in instances:
                inst_tokens = inst['tokens']
                L = inst['arg2_span'][1] - inst['arg2_span'][0]
                Pm = (inst['arg1_span'][0] + inst['arg2_span'][0]) // 2 + 1
                Ps = max(Pm - (window_length + 2), 0)
                Pe = min(Pm + (window_length + 2 + L), len(inst_tokens))
                tokens.append(inst_tokens[Ps:Pe])
            return_dict = self.tokenizer(
                tokens,
                is_split_into_words=True, return_tensors='pt',
                padding=True, truncation=True
            )

        if not self.ignore_labels:
            return_dict['labels'] = torch.tensor([self.label2id[inst['label']] for inst in instances]).long()
        else:
            return_dict['labels'] = torch.tensor([-1 for inst in instances]).long()

        inst_info_dict = {
            'sent_ids': [inst['sent_id'] for inst in instances],
            'doc_ids': [inst['doc_id'] for inst in instances],
            'arg1': [inst['arg1_span'] for inst in instances],
            'arg2': [inst['arg2_span'] for inst in instances]
        }

        return return_dict, inst_info_dict

    def generate_empty_predictions(self):
        if self.from_preprocessed:
            raise Exception("The dataset needs to be loaded from output data. Use preprocessed2output() first.")

        new_instances = []
        for inst in self.instances:
            new_instances.append({
                'doc_id': inst['doc_id'],
                'sent_id': inst['sent_id'],
                'tokens': inst['tokens'],
                'graph': {
                    'entities': inst['graph']['entities'].copy(),
                    'triggers': inst['graph']['triggers'].copy(),
                    'relations': inst['graph']['relations'].copy(),
                    'roles': []
                }
            })

        self.predictions = new_instances

    def add_predictions(self, inst_info_dict, predictions, coefs=None):
        if self.from_preprocessed:
            raise Exception("The dataset needs to be loaded from output data.")

        if not hasattr(self, 'predictions'):
            raise Exception(
                "The dataset needs to initialize predictions. Use generate_empty_predictions() before calling add_predictions().")

        batch = zip(*inst_info_dict.values(), predictions, coefs) if coefs else zip(*inst_info_dict.values(),
                                                                                    predictions,
                                                                                    [None] * len(predictions))

        for sent_id, doc_id, event_type, trigger, argument, label, coef in batch:
            inst_idx = self.instance2id[sent_id]
            label = self.id2label[label]
            inst = self.predictions[self.instance2id[sent_id]]
            assert inst['sent_id'] == sent_id
            coef = coef if coef else 1.0

            # Find event id
            event_id = -1
            for i, event_trigger in enumerate(inst["graph"]["triggers"]):
                if event_trigger[0] == trigger[0] and event_trigger[1] == trigger[1]:
                    event_id = i
                    break

            if event_id == -1:
                print('Event not found!')
                continue

            # Find arg id
            arg_id = -1
            for i, ent in enumerate(inst["graph"]["entities"]):
                if ent[0] == argument[0] and ent[1] == argument[1]:
                    arg_id = i
                    break

            if arg_id == -1:
                print('Entity not found!')
                continue

            # Find role id
            role_id = -1
            for i, rol in enumerate(inst["graph"]["roles"]):
                if rol[0] == event_id and rol[1] == arg_id:
                    role_id = i
                    break

            # if role_id == -1:
            #    continue

            if label != 'O':
                inst['graph']['roles'].append([event_id, arg_id, label, coef])

    def save_output(self, file_path):
        if self.from_preprocessed:
            raise Exception("The dataset needs to be loaded from output data.")

        # if not hasattr(self, 'predictions'):
        #    with open(file_path, 'wt') as out_f:
        #        for inst in self.instances:
        #            out_f.write(f"{json.dumps(inst)}\n")
        # else:
        with open(file_path, 'wt') as out_f:
            for inst in self.predictions:
                out_f.write(f"{json.dumps(inst)}\n")


def test_ACERelationClassificationDataset():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    new_tokens = ['<e1s>', '<e1e>', '<e2s>', '<e2e>']
    tokenizer.add_tokens(new_tokens)

    dataset = ACERelationClassificationDataset(file_path='./datasets/ace-e+/test.oneie.json', tokenizer=tokenizer,
                                               from_preprocessed=True)
    dataset.preprocessed2output()

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn,
                            shuffle=True)

    step, tp, p, r, total, total_loss = 0, 0, 0, 0, 0, 0
    for batch, inst_info in dataloader:
        pprint(batch)
        pprint(inst_info)
        break


if __name__ == "__main__":
    test_ACERelationClassificationDataset()
