import torch
import torch.nn.functional as F
import numpy as np
import time

@ torch.no_grad()
def decoding_default(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Default decoding function from LLaDA paper
    '''
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    # attention_mask: bs, 1, seq_len, seq_len, all true
    attention_mask = torch.ones(1, 1, x.shape[1], x.shape[1], dtype=torch.bool).to(model.device)
    
    position_ids = torch.arange(x.shape[1], device=x.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask, position_ids=position_ids).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x, steps * num_blocks

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
        
     
@torch.no_grad()
def decoding_wino(model, prompt, gen_length=128, block_length=128, temperature=0., mask_id=126336, threshold=0.6, threshold_back=0.9):

    device = model.device
    x_block = torch.full((1, prompt.shape[1] + gen_length + block_length), mask_id, dtype=torch.long).to(model.device)
    x_block[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x_block != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    step = 0
    

    for num_block in range(num_blocks):
        block_step = 0
        mask_index_block = (x_block == mask_id) # b, l
        mask_index_block[:, prompt.shape[1] + (num_block + 1) * block_length:] = False
        
        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:,  -block_length:] = ~mask_index_block[:, prompt.shape[1] + num_block* block_length: prompt.shape[1] + (num_block + 1) * block_length]
        position_ids = torch.cat([torch.arange(prompt.shape[1] + gen_length, device=device), torch.arange(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length, device=device)])
        attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = torch.ones(block_length, block_length, dtype=torch.bool).to(device)
        attention_mask[:, :, -block_length:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = ~torch.eye(block_length, dtype=torch.bool).to(device)
        last_accept = 30
        while mask_index_block.any():
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits # b, l, vocab_size
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_block_shift_left[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_block_shift_left]

            p = F.softmax(logits.to(torch.float64), dim=-1) # b, l, vocab_size
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            x0 = torch.where(mask_index_block, x0, x_block) # replace the masked tokens with the predicted tokens
            confidence = torch.where(mask_index_block, x0_p, -np.inf) # keep the confidence of the masked tokens
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)
            

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                # get top max_accept tokens
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            
            # always transfer the max confidence token
            else:
                if not transfer_index.any():
                    max_confidence_index = torch.argmax(confidence)
                    transfer_index.view(-1)[max_confidence_index] = True
            x_block[transfer_index] = x0[transfer_index]
            
            num_accept = transfer_index.sum()
            
            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    confidence_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                    _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)
            
            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            block_step += 1
            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
            unmask_index_block[transfer_index_shift] = True
            unmask_index_block[remask_index] = False
            last_accept = num_accept

        step += block_step

    return x_block[:, :prompt.shape[1] + gen_length], step


def construct_remix_soft_embedding(posterior, embedding_weight, mask_id=126336,
                                    beta_mix=0.5, top_k=50):
    """Construct the continuous input used by ReMix for one or more positions.

    This is algebraically equivalent to ``decoding_remix`` without allocating a
    dense selected-probability tensor: unselected posterior mass is assigned to
    the MASK embedding, and the result is mixed with the logical MASK input.
    """
    probs = posterior.to(dtype=embedding_weight.dtype)
    k = min(probs.shape[-1], top_k)
    top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
    cumulative = torch.cumsum(top_probs, dim=-1)
    alpha = (2 * top_probs[..., 0]).clamp(min=0.2, max=0.9)
    selected = cumulative <= alpha.unsqueeze(-1)
    selected[..., 0] = True
    selected_probs = torch.where(selected, top_probs, torch.zeros_like(top_probs))
    selected_embeddings = embedding_weight[top_indices]
    selected_mass = selected_probs.sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
    mask_embedding = embedding_weight[mask_id]
    soft_embedding = (selected_probs.unsqueeze(-1) * selected_embeddings).sum(dim=-2)
    soft_embedding = soft_embedding + (1.0 - selected_mass) * mask_embedding
    return (1.0 - beta_mix) * mask_embedding + beta_mix * soft_embedding


def _construct_remix_soft_embedding_dense_reference(posterior, embedding_weight,
                                                     mask_id=126336,
                                                     beta_mix=0.5, top_k=50):
    """Literal dense ReMix construction, used only by the smoke parity test."""
    probs = posterior.to(dtype=embedding_weight.dtype)
    k = min(probs.shape[-1], top_k)
    top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
    cumulative = torch.cumsum(top_probs, dim=-1)
    alpha = (2 * top_probs.max(dim=-1).values).clamp(min=0.2, max=0.9)
    valid = cumulative <= alpha.unsqueeze(-1)
    valid[..., 0] = True
    selected_probs = torch.zeros_like(probs)
    selected_probs.scatter_(-1, top_indices,
                            torch.where(valid, top_probs, torch.zeros_like(top_probs)))
    selected_mass = selected_probs.sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
    mask_embedding = embedding_weight[mask_id]
    soft_embedding = selected_probs @ embedding_weight + (1.0 - selected_mass) * mask_embedding
    return (1.0 - beta_mix) * mask_embedding + beta_mix * soft_embedding


def _select_wino_suspicious(confidence_back, num_accept, last_accept, threshold_back):
    """The suspicious-HARD selection in decoding_wino, kept as an exact helper."""
    if num_accept > 1:
        remask_index = confidence_back < threshold_back
        if remask_index.sum() >= last_accept:
            num_remask = last_accept - 1
            confidence_flat = confidence_back.view(-1)
            temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
            _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
            temp_mask[indices] = True
            remask_index = temp_mask.view(confidence_back.shape)
    else:
        remask_index = torch.zeros_like(confidence_back, dtype=torch.bool)
    return remask_index


def _posterior_entropy(probabilities):
    probabilities = probabilities.to(torch.float64)
    return -(probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(dim=-1)


def _row_jsd(p, q):
    """Exact row-wise Jensen-Shannon divergence in nats."""
    p = p.to(torch.float64)
    q = q.to(torch.float64)
    midpoint = 0.5 * (p + q)
    tiny = torch.finfo(torch.float64).tiny
    return 0.5 * (
        (p * (p.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log())).sum(dim=-1)
        + (q * (q.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log())).sum(dim=-1)
    )


def _row_three_view_jsd(p0, p1, p2):
    """Generalized JSD for three posterior views, in nats."""
    p0 = p0.to(torch.float64)
    p1 = p1.to(torch.float64)
    p2 = p2.to(torch.float64)
    midpoint = (p0 + p1 + p2) / 3.0
    tiny = torch.finfo(torch.float64).tiny
    log_midpoint = midpoint.clamp_min(tiny).log()
    return (
        (p0 * (p0.clamp_min(tiny).log() - log_midpoint)).sum(dim=-1)
        + (p1 * (p1.clamp_min(tiny).log() - log_midpoint)).sum(dim=-1)
        + (p2 * (p2.clamp_min(tiny).log() - log_midpoint)).sum(dim=-1)
    ) / 3.0


@torch.no_grad()
def _decoding_wino_revision(model, prompt, revision_mode, gen_length=128,
                            block_length=128, temperature=0., mask_id=126336,
                            threshold=0.6, threshold_back=0.9,
                            beta_mix=0.5, return_diagnostics=False,
                            verify_soft_parity=False,
                            stability_probe=False):
    """Instrumented WINO with a selectable suspicious-token revision action.

    ``remask`` reproduces WINO's H->MASK action. ``soft`` changes only that
    action to H->S and resolves S on the immediately following normal forward.
    ``soft_top1`` uses the same lifecycle but commits the current verification
    posterior top-1 after a pass. ``soft_v4`` keeps the one-forward lifecycle:
    it restores the old HARD above ``threshold_back``, otherwise commits the
    current top-1 above ``threshold``, and falls back to MASK. ``soft_v5``
    retains V4's same-identity early recommit but sends every different-identity
    alternative to MASK. ``soft_v6`` retains only V4's different-identity
    correction and sends same-identity probabilities below ``threshold_back``
    to MASK. ``soft_renew``
    renews the current-posterior
    Soft embedding while the old HARD probability is between WINO's forward
    and backward thresholds. ``none`` leaves suspicious HARD tokens unchanged
    and is provided only as a lightweight ablation.
    """
    if revision_mode not in {
            'remask', 'soft', 'soft_top1', 'soft_v4', 'soft_v5', 'soft_v6',
            'soft_renew', 'none'}:
        raise ValueError(f"Unknown revision mode: {revision_mode}")

    device = model.device
    x_block = torch.full((1, prompt.shape[1] + gen_length + block_length),
                         mask_id, dtype=torch.long, device=device)
    x_block[:, :prompt.shape[1]] = prompt.clone()
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    step = 0
    embedding_weight = model.model.transformer.wte.weight

    diagnostics = {
        'revision_mode': revision_mode,
        'nfe': 0,
        'decoding_rounds': 0,
        'num_suspicious_hard_tokens': 0,
        'num_h_to_mask_revisions': 0,
        'num_h_to_soft_revisions': 0,
        'num_h_to_s_to_h': 0,
        'num_h_to_s_to_m': 0,
        'num_s_to_s': 0,
        'num_s_to_h': 0,
        'num_s_to_m': 0,
        'num_stalled_soft_to_mask_fallbacks': 0,
        'soft_lifetimes': [],
        'unique_revised_positions': [],
        'before_revision_confidences': [],
        'before_revision_entropies': [],
        'after_revision_confidences': [],
        'after_revision_entropies': [],
        'round_trace': [],
        'revision_events': [],
        'remask_identity_events': [],
        'soft_embeddings_consumed': 0,
        'soft_parity_max_abs_differences': [],
        'soft_parity_mean_abs_differences': [],
        'unresolved_soft_states': 0,
    }
    unique_revised_positions = set()
    next_event_id = 0

    for num_block in range(num_blocks):
        block_step = 0
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        tail_start = x_block.shape[1] - block_length
        mask_index_block = (x_block == mask_id)
        mask_index_block[:, block_end:] = False

        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:, -block_length:] = ~mask_index_block[:, block_start:block_end]
        position_ids = torch.cat([
            torch.arange(prompt.shape[1] + gen_length, device=device),
            torch.arange(block_start, block_end, device=device),
        ])
        attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1],
                                    dtype=torch.bool, device=device)
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = True
        attention_mask[:, :, -block_length:, block_start:block_end] = ~torch.eye(
            block_length, dtype=torch.bool, device=device)

        soft_mask = torch.zeros_like(mask_index_block)
        soft_embeddings = torch.zeros(
            (*x_block.shape, embedding_weight.shape[-1]),
            dtype=embedding_weight.dtype, device=device)
        pending_events = {}
        pending_remask_events = {}
        soft_embedding_history = {}
        # Probe-only tensors live for at most one round and are never used by
        # decoding decisions.  Float32 keeps the diagnostic memory bounded.
        probe_creation_posteriors = {}
        pending_global_impact = None
        anchor_streak = {}
        last_accept = 30

        while mask_index_block.any():
            soft_at_start = soft_mask.clone()
            if soft_at_start.any():
                assert torch.all(mask_index_block[soft_at_start])
                local_soft = soft_at_start[:, block_start:block_end]
                verifier_soft = torch.zeros_like(unmask_index_block)
                verifier_soft[:, -block_length:] = local_soft
                assert torch.all(unmask_index_block[verifier_soft])

            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            if revision_mode in {
                    'soft', 'soft_top1', 'soft_v4', 'soft_v5', 'soft_v6',
                    'soft_renew'} and soft_at_start.any():
                inputs_embeds = embedding_weight[x_block]
                inputs_embeds[soft_at_start] = soft_embeddings[soft_at_start]
                assert torch.equal(inputs_embeds[soft_at_start], soft_embeddings[soft_at_start])
                logits = model(None, inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask,
                               position_ids=position_ids).logits
                diagnostics['soft_embeddings_consumed'] += int(soft_at_start.sum().item())
            else:
                logits = model(x_block, attention_mask=attention_mask,
                               position_ids=position_ids).logits
            diagnostics['nfe'] += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_block_shift_left[:, block_start:block_end] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_block_shift_left]

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            if stability_probe and pending_global_impact is not None:
                observer_positions = pending_global_impact['observer_positions']
                if observer_positions:
                    after_observer = p[0, observer_positions].to(torch.float32)
                    global_jsd = _row_jsd(
                        pending_global_impact['observer_posteriors'], after_observer)
                    before_top1 = pending_global_impact['observer_top1']
                    after_top1 = torch.argmax(after_observer, dim=-1)
                    global_values = {
                        'global_jsd_mean': float(global_jsd.mean().item()),
                        'global_jsd_median': float(global_jsd.median().item()),
                        'global_jsd_normalized_mean': float(
                            (global_jsd.mean() / np.log(2.0)).item()),
                        'observer_count': len(observer_positions),
                        'observer_top1_flip_rate': float(
                            (before_top1 != after_top1).to(torch.float32).mean().item()),
                    }
                else:
                    global_values = {
                        'global_jsd_mean': None,
                        'global_jsd_median': None,
                        'global_jsd_normalized_mean': None,
                        'observer_count': 0,
                        'observer_top1_flip_rate': None,
                    }

                anchor_positions = pending_global_impact['anchor_positions']
                if anchor_positions:
                    anchor_verifiers = [
                        tail_start + (position - block_start)
                        for position in anchor_positions]
                    anchor_tokens = torch.tensor(
                        pending_global_impact['anchor_tokens'], device=device,
                        dtype=torch.long)
                    after_anchor = p[0, anchor_verifiers].gather(
                        1, anchor_tokens[:, None]).squeeze(1)
                    before_anchor = pending_global_impact['anchor_probabilities'].to(
                        after_anchor.dtype)
                    tiny = torch.finfo(torch.float64).tiny
                    damage = (before_anchor.clamp_min(tiny).log()
                              - after_anchor.clamp_min(tiny).log())
                    global_values.update({
                        'anchor_count': len(anchor_positions),
                        'anchor_damage_mean': float(damage.mean().item()),
                        'anchor_damage_median': float(damage.median().item()),
                        'anchor_damage_positive_rate': float(
                            (damage > 0).to(torch.float32).mean().item()),
                    })
                else:
                    global_values.update({
                        'anchor_count': 0,
                        'anchor_damage_mean': None,
                        'anchor_damage_median': None,
                        'anchor_damage_positive_rate': None,
                    })
                for event_id in pending_global_impact['event_ids']:
                    diagnostics['revision_events'][event_id].update(global_values)
                pending_global_impact = None

            if stability_probe:
                hard_positions = torch.nonzero(
                    ~mask_index_block[0, block_start:block_end],
                    as_tuple=False).flatten().tolist()
                hard_positions = [block_start + offset for offset in hard_positions]
                current_hard_set = set(hard_positions)
                for position in list(anchor_streak):
                    if position not in current_hard_set:
                        anchor_streak.pop(position, None)
                for position in hard_positions:
                    verifier = tail_start + (position - block_start)
                    token_id = int(x_block[0, position].item())
                    verifier_top1 = int(torch.argmax(p[0, verifier]).item())
                    verifier_support = float(p[0, verifier, token_id].item())
                    if verifier_top1 == token_id and verifier_support >= threshold_back:
                        anchor_streak[position] = anchor_streak.get(position, 0) + 1
                    else:
                        anchor_streak[position] = 0

            resolved_positions = []
            renewed_positions = []
            identity_correction_event_ids = []
            if revision_mode in {
                    'soft', 'soft_top1', 'soft_v4', 'soft_v5', 'soft_v6',
                    'soft_renew'} and soft_at_start.any():
                soft_positions_at_start = torch.nonzero(
                    soft_at_start[0], as_tuple=False).flatten().tolist()
                if stability_probe and soft_positions_at_start:
                    p0_batch = torch.stack([
                        probe_creation_posteriors[position]
                        for position in soft_positions_at_start])
                    p1_batch = p[0, soft_positions_at_start].to(torch.float32)
                    verifier_positions = [
                        tail_start + (position - block_start)
                        for position in soft_positions_at_start]
                    p2_batch = p[0, verifier_positions].to(torch.float32)
                    d_temp = _row_jsd(p0_batch, p2_batch)
                    d_view = _row_jsd(p1_batch, p2_batch)
                    d_three = _row_three_view_jsd(p0_batch, p1_batch, p2_batch)
                    top2_p0 = torch.topk(p0_batch, k=2, dim=-1)
                    top2_p1 = torch.topk(p1_batch, k=2, dim=-1)
                    top2_p2 = torch.topk(p2_batch, k=2, dim=-1)
                    b_ids = top2_p2.indices[:, 0]
                    b_support = torch.stack([
                        p0_batch.gather(1, b_ids[:, None]).squeeze(1),
                        p1_batch.gather(1, b_ids[:, None]).squeeze(1),
                        p2_batch.gather(1, b_ids[:, None]).squeeze(1),
                    ], dim=1)
                    # Exact 1-based rank of the round-t candidate b under the
                    # immediately preceding shadow posterior P0. Ties share
                    # the best rank; this diagnostic never affects decoding.
                    candidate_prev_rank = 1 + (
                        p0_batch > b_support[:, 0:1]).sum(dim=-1)
                    margins = torch.stack([
                        top2_p0.values[:, 0] - top2_p0.values[:, 1],
                        top2_p1.values[:, 0] - top2_p1.values[:, 1],
                        top2_p2.values[:, 0] - top2_p2.values[:, 1],
                    ], dim=1)

                for soft_offset, absolute_position in enumerate(soft_positions_at_start):
                    verifier_position = tail_start + (absolute_position - block_start)
                    after_confidence = float(x0_p[0, verifier_position].item())
                    after_entropy = float(_posterior_entropy(p[0, verifier_position]).item())
                    event_id = pending_events[absolute_position]
                    event = diagnostics['revision_events'][event_id]
                    if stability_probe:
                        event.update({
                            'd_temp_jsd': float(d_temp[soft_offset].item()),
                            'd_view_jsd': float(d_view[soft_offset].item()),
                            'three_view_gjsd': float(d_three[soft_offset].item()),
                            'd_temp_jsd_normalized': float(
                                (d_temp[soft_offset] / np.log(2.0)).item()),
                            'd_view_jsd_normalized': float(
                                (d_view[soft_offset] / np.log(2.0)).item()),
                            'three_view_gjsd_normalized': float(
                                (d_three[soft_offset] / np.log(3.0)).item()),
                            'temporal_top1_stable': bool(
                                top2_p0.indices[soft_offset, 0]
                                == top2_p2.indices[soft_offset, 0]),
                            'three_view_top1_agreement': bool(
                                top2_p0.indices[soft_offset, 0]
                                == top2_p1.indices[soft_offset, 0]
                                == top2_p2.indices[soft_offset, 0]),
                            'candidate_support_p0': float(
                                b_support[soft_offset, 0].item()),
                            'candidate_support_p1': float(
                                b_support[soft_offset, 1].item()),
                            'candidate_support_p2': float(
                                b_support[soft_offset, 2].item()),
                            'candidate_min_support': float(
                                b_support[soft_offset].min().item()),
                            'candidate_min_margin': float(
                                margins[soft_offset].min().item()),
                            'candidate_prev_rank': int(
                                candidate_prev_rank[soft_offset].item()),
                        })
                    top1_probability, top1_token = torch.max(
                        p[0, verifier_position], dim=-1)
                    top1_token_id = int(top1_token.item())
                    top1_confidence = float(top1_probability.item())
                    if revision_mode in {'soft_v4', 'soft_v5', 'soft_v6'}:
                        event['current_top1_token_id'] = top1_token_id
                        event['p_old'] = after_confidence
                        event['p_top1'] = top1_confidence
                    if revision_mode == 'soft_v6':
                        # Both distributions already come from this normal WINO
                        # forward.  The original position consumes the SOFT
                        # embedding; the appended position is WINO's shadow
                        # verifier.  V6 continues to arbitrate with the latter.
                        soft_top1_probability, soft_top1_token = torch.max(
                            p[0, absolute_position], dim=-1)
                        soft_top1_token_id = int(soft_top1_token.item())
                        event['soft_top1_token_id'] = soft_top1_token_id
                        event['shadow_top1_token_id'] = top1_token_id
                        event['p_soft_b_soft'] = float(
                            soft_top1_probability.item())
                        event['p_shadow_b_soft'] = float(
                            p[0, verifier_position, soft_top1_token_id].item())
                        event['p_shadow_b_shadow'] = top1_confidence
                        event['soft_shadow_top1_agreement'] = (
                            soft_top1_token_id == top1_token_id)
                        event['v6_commit_posterior'] = 'shadow_verifier'
                    event.setdefault('verification_history', []).append({
                        'round': step + block_step,
                        'old_hard_confidence': after_confidence,
                        'posterior_entropy': after_entropy,
                    })
                    diagnostics['after_revision_confidences'].append(after_confidence)
                    diagnostics['after_revision_entropies'].append(after_entropy)
                    if after_confidence >= threshold_back:
                        pending_events.pop(absolute_position)
                        soft_embedding_history.pop(absolute_position, None)
                        if revision_mode == 'soft_top1':
                            new_token_id = int(torch.argmax(
                                p[0, verifier_position], dim=-1).item())
                            x_block[0, absolute_position] = new_token_id
                            event['new_token_id'] = new_token_id
                            event['same_token'] = new_token_id == event['token_id']
                        elif revision_mode in {'soft_v4', 'soft_v5', 'soft_v6'}:
                            new_token_id = event['original_hard_token_id']
                            x_block[0, absolute_position] = new_token_id
                            event['new_token_id'] = new_token_id
                            event['same_token'] = True
                            event['identity_changed'] = False
                            event['final_action'] = 'S->H(a)'
                            event['commit_reason'] = 'p_old>=threshold_back'
                        mask_index_block[0, absolute_position] = False
                        diagnostics['num_h_to_s_to_h'] += 1
                        diagnostics['num_s_to_h'] += 1
                        event['outcome'] = 'H->S->H'
                        event['final_transition'] = 'S->H'
                    elif (revision_mode == 'soft_v5'
                          and top1_token_id == event['original_hard_token_id']
                          and after_confidence >= threshold):
                        pending_events.pop(absolute_position)
                        new_token_id = event['original_hard_token_id']
                        x_block[0, absolute_position] = new_token_id
                        mask_index_block[0, absolute_position] = False
                        diagnostics['num_h_to_s_to_h'] += 1
                        diagnostics['num_s_to_h'] += 1
                        event['new_token_id'] = new_token_id
                        event['same_token'] = True
                        event['identity_changed'] = False
                        event['final_action'] = 'S->H(a)'
                        event['commit_reason'] = 'same_identity_early_recommit'
                        event['outcome'] = 'H->S->H'
                        event['final_transition'] = 'S->H'
                    elif (revision_mode in {'soft_v4', 'soft_v6'}
                          and top1_confidence >= threshold
                          and (revision_mode == 'soft_v4'
                               or top1_token_id != event['original_hard_token_id'])):
                        pending_events.pop(absolute_position)
                        new_token_id = top1_token_id
                        x_block[0, absolute_position] = new_token_id
                        # Downstream WINO verification must score the newly
                        # committed token, rather than retaining p(old H).
                        x0_p[0, verifier_position] = top1_probability
                        mask_index_block[0, absolute_position] = False
                        diagnostics['num_h_to_s_to_h'] += 1
                        diagnostics['num_s_to_h'] += 1
                        event['new_token_id'] = new_token_id
                        event['same_token'] = new_token_id == event['original_hard_token_id']
                        event['identity_changed'] = not event['same_token']
                        event['final_action'] = 'S->H(b)'
                        event['commit_reason'] = 'different_identity_correction'
                        event['outcome'] = 'H->S->H'
                        event['final_transition'] = 'S->H'
                        identity_correction_event_ids.append(event_id)
                    elif revision_mode == 'soft_renew' and after_confidence >= threshold:
                        renewed_embedding = construct_remix_soft_embedding(
                            p[0, verifier_position], embedding_weight,
                            mask_id=mask_id, beta_mix=beta_mix)
                        seen_embeddings = soft_embedding_history[absolute_position]
                        if any(torch.equal(renewed_embedding, seen_embedding)
                               for seen_embedding in seen_embeddings):
                            # With no finite round cap in WINO, an exactly
                            # unchanged bf16 Soft state is a decoding fixed
                            # point. Resolve it conservatively to MASK so the
                            # required zero-SOFT termination remains possible.
                            pending_events.pop(absolute_position)
                            x_block[0, absolute_position] = mask_id
                            mask_index_block[0, absolute_position] = True
                            unmask_index_block[0, verifier_position] = False
                            diagnostics['num_h_to_s_to_m'] += 1
                            diagnostics['num_s_to_m'] += 1
                            diagnostics['num_stalled_soft_to_mask_fallbacks'] += 1
                            event['outcome'] = 'H->S->M'
                            event['final_transition'] = 'S->M'
                            event['termination_reason'] = 'exact_soft_embedding_cycle'
                            soft_embedding_history.pop(absolute_position)
                        else:
                            soft_embeddings[0, absolute_position] = renewed_embedding
                            seen_embeddings.append(renewed_embedding.clone())
                            event['num_s_to_s_renewals'] += 1
                            diagnostics['num_s_to_s'] += 1
                            renewed_positions.append(absolute_position)
                            continue
                    else:
                        pending_events.pop(absolute_position)
                        soft_embedding_history.pop(absolute_position, None)
                        x_block[0, absolute_position] = mask_id
                        mask_index_block[0, absolute_position] = True
                        unmask_index_block[0, verifier_position] = False
                        diagnostics['num_h_to_s_to_m'] += 1
                        diagnostics['num_s_to_m'] += 1
                        event['outcome'] = 'H->S->M'
                        event['final_transition'] = 'S->M'
                        if revision_mode in {'soft_v4', 'soft_v5', 'soft_v6'}:
                            event['new_token_id'] = None
                            event['same_token'] = None
                            event['identity_changed'] = False
                            event['final_action'] = 'S->M'
                            event['commit_reason'] = 'mask_fallback'
                    event['after_confidence'] = after_confidence
                    event['after_entropy'] = after_entropy
                    event['resolved_round'] = step + block_step
                    event['lifetime'] = step + block_step - event['created_round']
                    event['soft_lifetime'] = event['lifetime']
                    if revision_mode in {'soft_v4', 'soft_v5', 'soft_v6'}:
                        assert event['lifetime'] == 1, (
                            'Soft Revision V4/V5 must resolve after exactly one normal forward')
                    if revision_mode == 'soft_v5' and event['final_transition'] == 'S->H':
                        assert event['new_token_id'] == event['original_hard_token_id'], (
                            'Soft Revision V5 must never change token identity')
                    if (revision_mode == 'soft_v6'
                            and event.get('commit_reason') == 'different_identity_correction'):
                        assert event['new_token_id'] != event['original_hard_token_id'], (
                            'Soft Revision V6 correction must change token identity')
                    diagnostics['soft_lifetimes'].append(event['lifetime'])
                    soft_mask[0, absolute_position] = False
                    if stability_probe:
                        probe_creation_posteriors.pop(absolute_position, None)
                    resolved_positions.append(absolute_position)

            x0 = torch.where(mask_index_block, x0, x_block)
            eligible_mask = mask_index_block & ~soft_at_start
            confidence = torch.where(eligible_mask, x0_p, -np.inf)
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)
            if revision_mode == 'soft_renew' and soft_mask.any():
                current_soft_verifier = torch.zeros_like(soft_mask)
                current_soft_verifier[:, -block_length:] = soft_mask[:, block_start:block_end]
                # Renewable SOFT positions remain verifier targets, but are no
                # longer committed HARD candidates for the suspicious-H detector.
                confidence_back[current_soft_verifier] = np.inf

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            elif not transfer_index.any() and eligible_mask.any():
                max_confidence_index = torch.argmax(confidence)
                transfer_index.view(-1)[max_confidence_index] = True
            x_block[transfer_index] = x0[transfer_index]
            if revision_mode == 'remask' and pending_remask_events:
                transferred_positions = torch.nonzero(
                    transfer_index[0], as_tuple=False).flatten().tolist()
                for absolute_position in transferred_positions:
                    if absolute_position not in pending_remask_events:
                        continue
                    event_id = pending_remask_events.pop(absolute_position)
                    event = diagnostics['remask_identity_events'][event_id]
                    new_token_id = int(x_block[0, absolute_position].item())
                    event['new_token_id'] = new_token_id
                    event['rehardening_round'] = step + block_step
                    event['same_token'] = new_token_id == event['original_token_id']
                    event['rounds_spent_remasked'] = (
                        event['rehardening_round'] - event['remask_round'])
            num_accept = transfer_index.sum()

            suspicious_index = _select_wino_suspicious(
                confidence_back, num_accept, last_accept, threshold_back)
            suspicious_shift = torch.zeros_like(suspicious_index)
            suspicious_shift[:, block_start:block_end] = suspicious_index[:, -block_length:]
            suspicious_positions = torch.nonzero(suspicious_shift[0], as_tuple=False).flatten().tolist()
            diagnostics['num_suspicious_hard_tokens'] += len(suspicious_positions)
            unique_revised_positions.update(suspicious_positions)

            before_confidences = []
            before_entropies = []
            for absolute_position in suspicious_positions:
                verifier_position = tail_start + (absolute_position - block_start)
                before_confidences.append(float(x0_p[0, verifier_position].item()))
                before_entropies.append(float(_posterior_entropy(p[0, verifier_position]).item()))
            diagnostics['before_revision_confidences'].extend(before_confidences)
            diagnostics['before_revision_entropies'].extend(before_entropies)

            if revision_mode == 'remask':
                for absolute_position in suspicious_positions:
                    assert absolute_position not in pending_remask_events
                    event_id = len(diagnostics['remask_identity_events'])
                    diagnostics['remask_identity_events'].append({
                        'event_id': event_id,
                        'block': num_block,
                        'position': absolute_position,
                        'original_token_id': int(x_block[0, absolute_position].item()),
                        'remask_round': step + block_step,
                    })
                    pending_remask_events[absolute_position] = event_id
                x_block[suspicious_shift] = mask_id
                mask_index_block[suspicious_shift] = True
                unmask_index_block[suspicious_index] = False
                diagnostics['num_h_to_mask_revisions'] += len(suspicious_positions)
            elif revision_mode in {
                    'soft', 'soft_top1', 'soft_v4', 'soft_v5', 'soft_v6',
                    'soft_renew'}:
                for event_offset, absolute_position in enumerate(suspicious_positions):
                    verifier_position = tail_start + (absolute_position - block_start)
                    assert x_block[0, absolute_position] != mask_id
                    compact_soft_embedding = construct_remix_soft_embedding(
                        p[0, verifier_position], embedding_weight,
                        mask_id=mask_id, beta_mix=beta_mix)
                    soft_embeddings[0, absolute_position] = compact_soft_embedding
                    if revision_mode == 'soft_renew':
                        soft_embedding_history[absolute_position] = [
                            compact_soft_embedding.clone()]
                    if (verify_soft_parity and
                            len(diagnostics['soft_parity_max_abs_differences']) < 8):
                        dense_soft_embedding = _construct_remix_soft_embedding_dense_reference(
                            p[0, verifier_position], embedding_weight,
                            mask_id=mask_id, beta_mix=beta_mix)
                        difference = (compact_soft_embedding - dense_soft_embedding).abs().float()
                        diagnostics['soft_parity_max_abs_differences'].append(float(difference.max().item()))
                        diagnostics['soft_parity_mean_abs_differences'].append(float(difference.mean().item()))
                    soft_mask[0, absolute_position] = True
                    mask_index_block[0, absolute_position] = True
                    assert unmask_index_block[0, verifier_position]
                    event = {
                        'event_id': next_event_id,
                        'block': num_block,
                        'created_round': step + block_step,
                        'position': absolute_position,
                        'token_id': int(x_block[0, absolute_position].item()),
                        'original_hard_token_id': int(x_block[0, absolute_position].item()),
                        'soft_start_round': step + block_step,
                        'num_s_to_s_renewals': 0,
                        'before_confidence': before_confidences[event_offset],
                        'before_entropy': before_entropies[event_offset],
                    }
                    diagnostics['revision_events'].append(event)
                    if stability_probe:
                        probe_creation_posteriors[absolute_position] = (
                            p[0, verifier_position].to(torch.float32).clone())
                    pending_events[absolute_position] = next_event_id
                    next_event_id += 1
                diagnostics['num_h_to_soft_revisions'] += len(suspicious_positions)

            if stability_probe and identity_correction_event_ids:
                excluded = set(suspicious_positions)
                excluded.update(resolved_positions)
                excluded.update(torch.nonzero(
                    transfer_index[0], as_tuple=False).flatten().tolist())
                observer_positions = torch.nonzero(
                    mask_index_block[0, block_start:block_end],
                    as_tuple=False).flatten().tolist()
                observer_positions = [
                    block_start + offset for offset in observer_positions
                    if block_start + offset not in excluded]
                stable_anchor_positions = [
                    position for position, streak in anchor_streak.items()
                    if streak >= 2
                    and not bool(mask_index_block[0, position].item())
                    and position not in excluded]
                anchor_verifiers = [
                    tail_start + (position - block_start)
                    for position in stable_anchor_positions]
                anchor_tokens = [
                    int(x_block[0, position].item())
                    for position in stable_anchor_positions]
                if anchor_verifiers:
                    anchor_token_tensor = torch.tensor(
                        anchor_tokens, device=device, dtype=torch.long)
                    anchor_probabilities = p[0, anchor_verifiers].gather(
                        1, anchor_token_tensor[:, None]).squeeze(1).clone()
                else:
                    anchor_probabilities = torch.empty(
                        0, device=device, dtype=torch.float64)
                pending_global_impact = {
                    'event_ids': identity_correction_event_ids,
                    'observer_positions': observer_positions,
                    'observer_posteriors': (
                        p[0, observer_positions].to(torch.float32).clone()
                        if observer_positions else torch.empty(
                            0, p.shape[-1], device=device, dtype=torch.float32)),
                    'observer_top1': (
                        torch.argmax(p[0, observer_positions], dim=-1)
                        if observer_positions else torch.empty(
                            0, device=device, dtype=torch.long)),
                    'anchor_positions': stable_anchor_positions,
                    'anchor_tokens': anchor_tokens,
                    'anchor_probabilities': anchor_probabilities,
                }

            mask_index_block[transfer_index] = False
            block_step += 1
            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, block_start:block_end]
            unmask_index_block[transfer_index_shift] = True
            # A SOFT-only lifecycle round can intentionally draft zero tokens.
            # WINO's original loop never has such a round because its fallback
            # always transfers one MASK.  Keep the preceding cap state here;
            # otherwise last_accept=0 would make the next WINO top-k cap k=-1.
            if num_accept > 0:
                last_accept = num_accept
            diagnostics['round_trace'].append({
                'round': step + block_step - 1,
                'block': num_block,
                'transferred_positions': torch.nonzero(transfer_index[0], as_tuple=False).flatten().tolist(),
                'suspicious_positions': suspicious_positions,
                'resolved_soft_positions': resolved_positions,
                'renewed_soft_positions': renewed_positions,
                'soft_count_after_round': int(soft_mask.sum().item()),
            })

        assert not soft_mask.any(), "block completed with unresolved SOFT positions"
        assert not pending_events, "SOFT event bookkeeping was not resolved"
        assert not pending_remask_events, "remasked HARD event was not re-hardened"
        assert not soft_embedding_history, "SOFT embedding history was not cleared"
        assert not probe_creation_posteriors, "probe posterior bookkeeping was not cleared"
        step += block_step

    diagnostics['decoding_rounds'] = step
    diagnostics['unique_revised_positions'] = sorted(unique_revised_positions)
    diagnostics['num_unique_revised_positions'] = len(unique_revised_positions)
    diagnostics['average_soft_lifetime'] = (
        float(np.mean(diagnostics['soft_lifetimes'])) if diagnostics['soft_lifetimes'] else 0.0)
    soft_count = diagnostics['num_h_to_soft_revisions']
    diagnostics['fraction_soft_recovered_to_hard'] = (
        diagnostics['num_h_to_s_to_h'] / soft_count if soft_count else 0.0)
    diagnostics['unresolved_soft_states'] = 0
    diagnostics['unresolved_remask_identity_events'] = 0
    output = x_block[:, :prompt.shape[1] + gen_length]
    return (output, step, diagnostics) if return_diagnostics else (output, step)


def decoding_wino_remask(model, prompt, **kwargs):
    """Instrumented form of the unchanged WINO H->MASK baseline."""
    return _decoding_wino_revision(model, prompt, revision_mode='remask', **kwargs)


def decoding_wino_soft_revision(model, prompt, **kwargs):
    """WINO detector with one-normal-forward ReMix-style SOFT revision."""
    return _decoding_wino_revision(model, prompt, revision_mode='soft', **kwargs)


def decoding_wino_soft_revision_v2(model, prompt, **kwargs):
    """Soft Revision V2: passing SOFT commits verifier-posterior top-1."""
    return _decoding_wino_revision(model, prompt, revision_mode='soft_top1', **kwargs)


def decoding_wino_soft_revision_v3(model, prompt, **kwargs):
    """Renewable Soft Revision using WINO's existing two thresholds."""
    return _decoding_wino_revision(model, prompt, revision_mode='soft_renew', **kwargs)


def decoding_wino_soft_revision_v4(model, prompt, **kwargs):
    """One-round Soft Revision with old-H/top-1/MASK arbitration."""
    return _decoding_wino_revision(model, prompt, revision_mode='soft_v4', **kwargs)


def decoding_wino_soft_revision_v5(model, prompt, **kwargs):
    """One-round Soft Revision with same-identity early recommit only."""
    return _decoding_wino_revision(model, prompt, revision_mode='soft_v5', **kwargs)


def decoding_wino_soft_revision_v6(model, prompt, **kwargs):
    """One-round Soft Revision with different-identity correction only."""
    return _decoding_wino_revision(model, prompt, revision_mode='soft_v6', **kwargs)


def decoding_wino_no_revision(model, prompt, **kwargs):
    """Optional ablation in which suspicious HARD tokens remain committed."""
    return _decoding_wino_revision(model, prompt, revision_mode='none', **kwargs)
