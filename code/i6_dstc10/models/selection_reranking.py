import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward, ModelOutput,
)
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput, BaseModelOutput, SequenceClassifierOutput,
)


class RobertaForReranking(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.reranking_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        candidate_positions=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        sequence_output = outputs[0]

        logits = self.reranking_outputs(sequence_output).squeeze(axis=-1)
        
        total_loss = None
        candidate_logits = torch.gather(logits, -1, candidate_positions)
        if candidate_positions is not None and labels is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(candidate_logits, labels)

        if not return_dict:
            output = outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=candidate_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )