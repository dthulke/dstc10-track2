import torch
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class BartForAutoregressiveEncoder(BartForConditionalGeneration):
  def __init__(self, config):
    super().__init__(config)

  def forward(
      self,
      input_ids=None,
      labels=None,
      negative_labels=None,
      **model_kwargs,
  ):
    target_output = super(BartForAutoregressiveEncoder, self).forward(
      input_ids=input_ids,
      labels=labels,
      **model_kwargs,
    )

    # Cross entropy loss of labels
    loss = target_output.loss

    # If we have negative labels - add additional sequence level loss
    if negative_labels is not None and negative_labels.shape[1] > 0:
      # Shape of negative_labels: (batch, num_negatives, time)
      batch_size, num_negatives, max_length = negative_labels.shape

      loss_fct = CrossEntropyLoss(reduction='none')

      encoder_outputs = (
        target_output.encoder_last_hidden_state, target_output.encoder_hidden_states, target_output.encoder_attentions
      )

      denominator_losses = torch.zeros((batch_size, num_negatives+1), device=self.device)
      for i in range(num_negatives):
        current_negative_labels = negative_labels[:, i, :].contiguous()
        negatives_output = super(BartForAutoregressiveEncoder, self).forward(
          labels=current_negative_labels,
          input_ids=input_ids,
          encoder_outputs=encoder_outputs,
          use_cache=False,
        )
        negative_loss = loss_fct(negatives_output.logits.view(-1, self.config.vocab_size), current_negative_labels.view(-1)).view(batch_size, -1).sum(-1)
        denominator_losses[:, i] = negative_loss

      positive_loss = loss_fct(target_output.logits.view(-1, self.config.vocab_size), labels.view(-1)).view(batch_size, -1).sum(-1)
      # -log(p(...))
      denominator_losses[:, -1] = positive_loss
      loss = positive_loss + torch.logsumexp(-denominator_losses, dim=-1)
      loss = 0.5 * target_output.loss + 0.5 * loss.mean()

    return Seq2SeqLMOutput(
      loss=loss,
      logits=target_output.logits,
      past_key_values=target_output.past_key_values,
      decoder_hidden_states=target_output.decoder_hidden_states,
      decoder_attentions=target_output.decoder_attentions,
      cross_attentions=target_output.cross_attentions,
      encoder_last_hidden_state=target_output.encoder_last_hidden_state,
      encoder_hidden_states=target_output.encoder_hidden_states,
      encoder_attentions=target_output.encoder_attentions,
    )
