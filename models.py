
from torch import nn
import transformers
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
import random


class AttentionClassifier(nn.Module):
    """
    Neural/Contradiction detection attention layer
    """
    def __init__(self, dimensionality, dropout, dropout_type):
        super(AttentionClassifier,self).__init__()

        self.linear1 = torch.nn.Linear(dimensionality, dimensionality)
        self.linear2 = torch.nn.Linear(dimensionality, 1)
        self.tanh = torch.tanh
        self.linear3 = torch.nn.Linear(1, 1)
        self.sig = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.eval = True
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def eval_(self):
        self.eval = True


    def train(self):
        self.eval = False


    def forward(
              self, 
              rep: torch.tensor, 
              logit: torch.tensor,
              consec_segments_in_span) -> dict:
        """
        Forward pass for neutral/contradiction detection attention layers

        Args:
            rep: CLS representations of all spans (span no x CLS dimensions)
            logit: logit for all spans (span no x 1)
            consec_segments_in_span: which spans contain consec. segments

        Returns:
            output_val: dictionary, containing:
                'sent_output': probability of neutral/cont. for sentence
                'att_weights': normalized attention weights
                'att_unnorm': unnormalzed attention weights
                'dropout': dropout mask applied 
        """

        # batch_size x seq_len x dim:
        val = self.linear1(rep)
        val = self.tanh(val)

        # batch_size x seq_len x 1:
        val = self.linear2(val)
        val = self.sig(val)

        sum_val = torch.sum(val)
        att_unnorm = val
        inv_sum_val = 1/sum_val

        # batch_size x seq_len x 1:
        att_weights = val*inv_sum_val

        dropout_mask = torch.ones(att_weights.shape).to(self.device)
        
        if not self.eval:
            
            if self.dropout_type == 'standard':
                dropout_mask = self.standard_dropout(dropout_mask) 
            elif self.dropout_type == 'overlap':
                dropout_mask = self.overlap_dropout(
                        dropout_mask, 
                        consec_segments_in_span)

            att_weights = att_weights * dropout_mask
            att_unnorm = att_unnorm * dropout_mask

            new_sum = torch.sum(att_weights)
            att_weights = att_weights / new_sum

        #batch_size x dimensions
        updated_rep = torch.einsum('jk, jm -> k', [logit, att_weights])
        output_val = self.linear3(updated_rep)
        output_val = self.sig(output_val)

        # Preparing dictionary of outputs
        output_dict = {}
        output_dict['sent_output'] = output_val
        output_dict['att_weights'] = att_weights
        output_dict['att_unnorm'] = att_unnorm
        output_dict['dropout'] = dropout_mask

        return output_dict


    def standard_dropout(
            self, 
            dropout_mask: torch.tensor):
        """
        Random applies dropout to spans

        Args:
            dropout_mask: mask before dropout (all 1s)

        Returns:
            dropout_mask: updated mask when applying dropout
        """

        number_spans = dropout_mask.shape[0]
        spans_dropped_out = []

        # We allow dropout on the spans in the attention layer
        for span_no in range(number_spans):
            if random.choices([True, False],
                    [self.dropout, 1-self.dropout])[0] == True:
                spans_dropped_out.append(span_no)

        # We can't apply drop-out to every span
        if number_spans == len(spans_dropped_out):
            span_remove_dropout = random.randint(0,len(spans_dropped_out)-1)
            spans_dropped_out.remove(span_remove_dropout)

        for span_no in spans_dropped_out:
            dropout_mask[span_no] = 0

        return dropout_mask


    def overlap_dropout(
            self,
            dropout_mask: torch.tensor,
            overlap_list: list):
        """
        Applies dropout to all multi-segment spans, or to none at all

        Args:
            dropout_mask: mask before dropout (all 1s)
            overlap_list: How many consec. multi segments in the span

        Returns:
            dropout_mask: updated mask when applying dropout  
        """

        number_spans = dropout_mask.shape[0]
        spans_dropped_out = []

        # We allow dropout on the spans in the attention layer
        if random.choices([True, False],
                [self.dropout, 1-self.dropout])[0] \
                        == True:
            for span_no in range(number_spans):
                    if overlap_list[span_no] != 0:
                        spans_dropped_out.append(span_no)

            for span_no in spans_dropped_out:
                dropout_mask[span_no] = 0
        
        return dropout_mask


class LogicModel(nn.Module):
    """
    Our Logic NLI model
    """
    def __init__(self, dimensionality, model_type, span_dropout, dropout_type):
        super(LogicModel,self).__init__()

        self.attention_cont = AttentionClassifier(
                dimensionality, 
                span_dropout,
                dropout_type)
        self.attention_neutral = AttentionClassifier(
                dimensionality, 
                span_dropout,
                dropout_type)

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_type,
            output_attentions=True,
            output_hidden_states=True,
            num_labels=2)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_type = model_type


    def forward(
            self, 
            batch: dict, 
            obs_no: int, 
            span_mask_list_all: dict, 
            esnli_spans: dict) -> dict:
        """
        We create batch input from different span masks, and pass through model
        
        Args:
            batch: batch input into model (input ids, mask and token_type_ids)
            obs_no: observation number in minibatch to consider
            span_mask_list_all: list of all spans for each input ID
            esnli_spans: list of all e-snli spans for each input ID
        Returns:
            outputs: sentence outputs for each class, with unnormalized 
                attention, which spans to supervise, and the labels

        """

        # Find SEP token indices
        if self.model_type[0:4] == 'bert':
            sep_toks = torch.where(batch['input_ids'][obs_no] == 102)[0]
        elif self.model_type[0:9] == 'microsoft':
            sep_toks = torch.where(batch['input_ids'][obs_no] == 2)[0]
        
        # Find list of masks for each span
        obs_ids_no_padding = batch['input_ids'][obs_no][:sep_toks[1]+1]
        span_mask_list = span_mask_list_all[str(
            list(np.array(obs_ids_no_padding.cpu())))]

        # Find the eSNLI span mask for the observation
        if esnli_spans is not None:
            span_mask_esnli = esnli_spans[str(
                    list(np.array(obs_ids_no_padding.cpu())))]
        else:
            span_mask_esnli = None

        # We pass each span through BERT (with spans created from masking)
        all_spans_cls, all_spans_logits, span_dict = self.encode_all_spans(
                    batch,
                    span_mask_list,
                    span_mask_esnli,
                    sep_toks,
                    obs_no)

        # We set the labels for our neutral and condiction detection layers
        if batch['label'][obs_no] == 0:
            neutral_label=0
            cont_label=0
        elif batch['label'][obs_no] == 1:
            neutral_label=1
            cont_label=0
        else:
            neutral_label=-999
            cont_label=1

        # We find the outputs from the neutral/contradiction detection layers
        outputs = self.neutral_and_cont_detection(
                    span_dict,
                    all_spans_logits,
                    all_spans_cls,
                    neutral_label=neutral_label,
                    cont_label=cont_label)

        return outputs


    def supervise_span_or_not(
        self,
        span_mask_esnli_dict: dict,
        span: list) -> dict:
        """
        We decide if each span should be supervised (from matching to eSNLI)

        Args:
            span_mask_esnli_dict: the eSNLI spans for the observation
            span: the span in question
        
        Returns:
            supervise_span_dict: whether the span should be supervised directly
        """

        if span_mask_esnli_dict is None:
            return False
    
        supervise_span_dict = {}

        expl_col_list = list(span_mask_esnli_dict.keys())

        for expl_col in expl_col_list:
            # We may not have eSNLI data for the dataset
            span_mask_esnli = span_mask_esnli_dict[expl_col]

            # Supervise the span if the eSNLI span is contained within it
            supervise_span = False
            if 1 in span_mask_esnli:
                idx_in_esnli_span = [idx_ for idx_ in \
                        range(len(span_mask_esnli)) \
                        if span_mask_esnli[idx_] == 1]

                supervise_span = True

                for idx_ in idx_in_esnli_span:
                    if span[idx_] != 1:
                        supervise_span = False
            supervise_span_dict[expl_col] = supervise_span

        return supervise_span_dict


    def encode_all_spans(
        self,
        batch: dict,
        span_mask_list: list,
        span_mask_esnli: list,
        sep_toks: torch.tensor,
        obs_no: int) -> (torch.tensor, torch.tensor, dict):
        """
        Pass each span through the BERT model to find logits and CLS reps.

        Args:
            batch: input ids, attention mask and token_type_ids
            span_mask_list: list of masks for each span (and how many sub-spans
                ... each of these is made up of)
            span_mask_esnli: the eSNLI span mask for the observation
            sep_toks: location of sep tokens
            obs_no: observation in minibatch being considered

        Returns:
            obs_cls: cls representations for each span
            obs_logits: logits for neutral and contradiction classes
            span_dict: dictonary with further information on each span

        """ 

        # We find spans for the hypothesis
        consec_segments_in_span = []
        supervise_span_or_not = []
        attention_mask_list = []
        
        attention_mask = torch.zeros(
                    len(span_mask_list),
                    sep_toks[1]+1, 
                    dtype = torch.long).to(self.device)

        for span_no in range(len(span_mask_list)):
         
            # We check which spans are created from smaller, consecutive spans
            consec_segments_in_span.append(span_mask_list[span_no][1])

            # Which spans do we supervise
            supervise_span = self.supervise_span_or_not(
                    span_mask_esnli,
                    span_mask_list[span_no][0])
            supervise_span_or_not.append(supervise_span)

            # Update our attention masks:                                       
            for token_idx, token_mask in enumerate(span_mask_list[span_no][0]): 
                if token_mask == 1:                                             
                    attention_mask[span_no,token_idx + sep_toks[0] + 1] = 1 

            # We attend to all of the premise
            attention_mask[span_no,:sep_toks[0].item()] = 1
            attention_mask[span_no,0] = 1
            attention_mask[span_no,sep_toks] = 1

            # Storing mask so I can output the spans in evaluation
            attention_mask_list.append(attention_mask[span_no,:])

        input_ids = batch['input_ids'][obs_no][:sep_toks[1]+1].repeat(
                len(span_mask_list), 1)
        token_type_ids = batch['token_type_ids'][obs_no][:sep_toks[1]+1].repeat(
                len(span_mask_list), 1)

        input_ids_no_padding = batch['input_ids'][obs_no][:sep_toks[1]+1]

        # We now use each attention mask together as a batch
        span_outputs_dict = self.encoder(
                input_ids, 
                attention_mask,
                token_type_ids, 
                return_dict=True)
            
        obs_logits = span_outputs_dict['logits']

        obs_cls = span_outputs_dict['hidden_states'][-1][:,0,:]

        span_dict = {
                'true_label': batch['label'][obs_no],
                'input_ids_no_padding': input_ids_no_padding,
                'attention_mask_list': attention_mask_list,
                'consec_segments_in_span': consec_segments_in_span,
                'supervise_span_or_not': supervise_span_or_not,
                }

        return obs_cls, obs_logits, span_dict


    def neutral_and_cont_detection(
        self,
        output_dict: dict,
        all_spans_logits: torch.tensor,
        all_spans_cls: torch.tensor,
        neutral_label: int,
        cont_label: int,
        ) -> dict:
        """
        Apply the neutral and contradiction detection attention layers

        Args:
            output_dict: span information
            all_spans_logits: logits for each span (for both classes)
            all_spans_cls: cls representation for each span
            neutral_label: neutral detection label
            cont_label: cont detection label

        Returns:
            output_dict: dict of outputs for NLI sentence pairs
        """

        neutral_output = self.attention_neutral(
                all_spans_cls, 
                all_spans_logits[:,0].unsqueeze(1),
                output_dict['consec_segments_in_span'])

        neutral_output['label'] = torch.tensor(
                [neutral_label]).to(self.device)

        cont_output = self.attention_cont(
                all_spans_cls, 
                all_spans_logits[:,1].unsqueeze(1),
                output_dict['consec_segments_in_span'])

        cont_output['label'] = torch.tensor(
                [cont_label]).to(self.device)

        output_dict.update({
                'neutral': neutral_output,
                'cont': cont_output})

        return output_dict


