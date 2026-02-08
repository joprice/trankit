'''
Borrowed from https://github.com/stanfordnlp/stanza/blob/master/stanza/models/common/crf.py
Date: 2021/01/06
'''
"""
CRF loss and viterbi decoding.
"""

from ..utils.base_utils import *


class CRFLoss(nn.Module):
    """
    Calculate log-space crf loss, given unary potentials, a transition matrix
    and gold tag sequences.
    """

    def __init__(self, num_tag, batch_average=True):
        super().__init__()
        self._transitions = nn.Parameter(torch.zeros(num_tag, num_tag))
        self._batch_average = batch_average  # if not batch average, average on all tokens

    def forward(self, inputs, masks, tag_indices):
        """
        inputs: batch_size x seq_len x num_tags
        masks: batch_size x seq_len
        tag_indices: batch_size x seq_len
        @return:
            loss: CRF negative log likelihood on all instances.
            transitions: the transition matrix
        """
        # TODO: handle <start> and <end> tags
        self.bs, self.sl, self.nc = inputs.size()
        unary_scores = self.crf_unary_score(inputs, masks, tag_indices)
        binary_scores = self.crf_binary_score(inputs, masks, tag_indices)
        log_norm = self.crf_log_norm(inputs, masks, tag_indices)
        log_likelihood = unary_scores + binary_scores - log_norm  # batch_size
        loss = torch.sum(-log_likelihood)
        if self._batch_average:
            loss = loss / self.bs
        else:
            total = masks.eq(0).sum()
            loss = loss / (total + 1e-8)
        return loss, self._transitions

    def crf_unary_score(self, inputs, masks, tag_indices):
        """
        @return:
            unary_scores: batch_size
        """
        flat_inputs = inputs.view(self.bs, -1)
        flat_tag_indices = tag_indices + \
                           torch.arange(self.sl, device=tag_indices.device).long().unsqueeze(0) * self.nc
        unary_scores = torch.gather(flat_inputs, 1, flat_tag_indices).view(self.bs, -1)
        unary_scores.masked_fill_(masks, 0)
        return unary_scores.sum(dim=1)

    def crf_binary_score(self, inputs, masks, tag_indices):
        """
        @return:
            binary_scores: batch_size
        """
        # get number of transitions
        nt = tag_indices.size(-1) - 1
        start_indices = tag_indices[:, :nt]
        end_indices = tag_indices[:, 1:]
        # flat matrices
        flat_transition_indices = start_indices * self.nc + end_indices
        flat_transition_indices = flat_transition_indices.view(-1)
        flat_transition_matrix = self._transitions.view(-1)
        binary_scores = torch.gather(flat_transition_matrix, 0, flat_transition_indices) \
            .view(self.bs, -1)
        score_masks = masks[:, 1:]
        binary_scores.masked_fill_(score_masks, 0)
        return binary_scores.sum(dim=1)

    def crf_log_norm(self, inputs, masks, tag_indices):
        """
        Calculate the CRF partition in log space for each instance, following:
            http://www.cs.columbia.edu/~mcollins/fb.pdf
        @return:
            log_norm: batch_size
        """
        start_inputs = inputs[:, 0, :]  # bs x nc
        rest_inputs = inputs[:, 1:, :]
        rest_masks = masks[:, 1:]
        alphas = start_inputs  # bs x nc
        trans = self._transitions.unsqueeze(0)  # 1 x nc x nc
        # accumulate alphas in log space
        for i in range(rest_inputs.size(1)):
            transition_scores = alphas.unsqueeze(2) + trans  # bs x nc x nc
            new_alphas = rest_inputs[:, i, :] + log_sum_exp(transition_scores, dim=1)
            m = rest_masks[:, i].unsqueeze(1).expand_as(new_alphas)  # bs x nc, 1 for padding idx
            # apply masks
            new_alphas = torch.where(m, alphas, new_alphas)
            alphas = new_alphas
        log_norm = log_sum_exp(alphas, dim=1)
        return log_norm


def viterbi_decode_batch(scores, trans, lengths):
    """Batched viterbi forward pass on GPU, backtrace on CPU.
    scores: (B, T, C) GPU tensor
    trans:  (C, C) GPU tensor
    lengths: list[int]
    Returns: list of tag-id lists (one per batch element)
    """
    B, T, C = scores.shape
    if B == 0:
        return []
    if C > 32767:
        raise ValueError(f"num_tags {C} exceeds int16 range")
    if len(lengths) != B:
        raise ValueError(f"lengths ({len(lengths)}) != batch size ({B})")
    if not all(l > 0 for l in lengths):
        raise ValueError("all lengths must be positive")
    if max(lengths) > T:
        raise ValueError(f"max length ({max(lengths)}) exceeds seq dim ({T})")
    lengths_t = torch.tensor(lengths, device=scores.device, dtype=torch.long)
    trans = trans.to(device=scores.device, dtype=scores.dtype)

    trellis = scores[:, 0].clone()                        # (B, C) running row
    final_scores = trellis.clone()                         # (B, C) per-sample final
    backpointers = torch.zeros(B, T, C, device=scores.device, dtype=torch.int16)

    for t in range(1, T):
        v = trellis.unsqueeze(2) + trans.unsqueeze(0)      # (B, C, C)
        trellis = scores[:, t] + v.max(dim=1).values       # (B, C)
        backpointers[:, t] = v.argmax(dim=1).to(torch.int16)
        # snapshot final scores for samples ending at this timestep
        mask = (lengths_t - 1 == t)                        # (B,)
        if mask.any():
            final_scores[mask] = trellis[mask]

    # handle L=1 case: final_scores is already set from clone of scores[:,0]
    # two transfers: backpointers (B,T,C int16) + final scores (B,C)
    bp_cpu = backpointers.cpu().numpy()
    final_cpu = final_scores.cpu().numpy()

    results = []
    for i in range(B):
        L = lengths[i]
        seq = [int(np.argmax(final_cpu[i]))]
        for t in range(L - 1, 0, -1):
            seq.append(int(bp_cpu[i, t, seq[-1]]))
        seq.reverse()
        results.append(seq)
    return results


def viterbi_decode(scores, transition_params):
    """
    Decode a tag sequence with viterbi algorithm.
    scores: seq_len x num_tags (numpy array)
    transition_params: num_tags x num_tags (numpy array)
    @return:
        viterbi: a list of tag ids with highest score
        viterbi_score: the highest score
    """
    trellis = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    trellis[0] = scores[0]

    for t in range(1, scores.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = scores[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var
