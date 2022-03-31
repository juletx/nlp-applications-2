import torch
import torch.nn as nn

#from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel


class RobertaForTupleClassification(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Tuple classification attributes
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.first_marker_token_id = config.first_marker_token_id
        self.second_marker_token_id = config.second_marker_token_id

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.criterion = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        device = input_ids.get_device() if input_ids.get_device() >= 0 else None
        sequence_output = outputs[0]

        if isinstance(self.first_marker_token_id, list):
            head_id = torch.zeros(input_ids.shape[0]).long().to(device)
            for token_id in self.first_marker_token_id:
                head_id += torch.argmax((input_ids == token_id).float(), dim=-1)
        else:
            head_id = torch.argmax((input_ids == self.first_marker_token_id).float(), dim=-1)
        if isinstance(self.second_marker_token_id, list):
            tail_id = torch.zeros(input_ids.shape[0]).long().to(device)
            for token_id in self.second_marker_token_id:
                tail_id += torch.argmax((input_ids == token_id).float(), dim=-1)
        else:
            tail_id = torch.argmax((input_ids == self.second_marker_token_id).float(), dim=-1)
        idx = torch.arange(sequence_output.shape[0])
        

        head = sequence_output[idx, head_id, :]     # head representation
        tail = sequence_output[idx, tail_id, :]     # tail representation

        x = torch.cat([head, tail], dim=-1)
        x = torch.relu(self.linear(x))

        # Classification
        x = self.dropout(x)
        x = self.classifier(x)

        if labels is not None:
            loss = self.criterion(x, labels)

            return (loss, x)
        else:
            return x
