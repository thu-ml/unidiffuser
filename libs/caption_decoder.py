import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import default_data_collator
from transformers import EarlyStoppingCallback

data_collator = default_data_collator
es = EarlyStoppingCallback(early_stopping_patience=5)
import json
import argparse
from typing import Union, Optional
from collections import OrderedDict


# %% model initial
class ClipCaptionModel(nn.Module):
    """
    """

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        : param tokens: (Tensor) [N x max_seq_len] eg. [4 X 33]
        : param prefix: (Tensor) [N x prefix_length x 768] eg. [4 x 77 x 768]
        : param mask: (Tensor) [N x (prefix_length + max_seq_len) x 768] eg. [4 x 110 x768]

        : attribute embedding_text: (Tensor) [N x max_seq_len x 768] eg. [4 x 33 x 768]
        : attribute embedding_cat: (Tensor) [N x (prefix_length + max_seq_len) x 768] eg. [4 x 110 x 768]
        """
        embedding_text = self.gpt.transformer.wte(tokens)
        hidden = self.encode_prefix(prefix)
        prefix = self.decode_prefix(hidden)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        if self.hidden_dim is not None:
            return out, hidden
        else:
            return out

    def encode_decode_prefix(self, prefix):
        return self.decode_prefix(self.encode_prefix(prefix))

    def __init__(self, prefix_length: int, hidden_dim=None):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        base_tokenizer.add_special_tokens(special_tokens_dict)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2', eos_token_id=base_tokenizer.eos_token_id)
        self.gpt.resize_token_embeddings(len(base_tokenizer))

        self.hidden_dim = hidden_dim
        self.encode_prefix = nn.Linear(768, hidden_dim) if hidden_dim is not None else nn.Identity()
        self.decode_prefix = nn.Linear(hidden_dim, 768) if hidden_dim is not None else nn.Identity()




def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def generate_beam(
        model,
        tokenizer,
        beam_size: int = 5,
        prompt=None,
        embed=None,
        entry_length=67,
        temperature=1.0,
        stop_token: str = '<|EOS|>',
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        # pbar = tqdm(range(entry_length))
        # pbar.set_description("generating text ...")
        for i in range(entry_length):
            # print(generated.shape)
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)], skip_special_tokens=True)
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    model.train()
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.0,
        stop_token: str = '<|EOS|>',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


class CaptionDecoder(object):
    def __init__(self, device, pretrained_path, hidden_dim=-1):
        if hidden_dim < 0:
            hidden_dim = None
        # tokenizer initialize
        eos = '<|EOS|>'
        special_tokens_dict = {'eos_token': eos}
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # model initialize
        feature_length = 77
        # modelFile = "assets/caption_decoder/coco_v2_latest.pt"
        self.caption_model = ClipCaptionModel(feature_length, hidden_dim=hidden_dim)
        # print("Load Model...")
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt.items():
            new_k = k[7:]
            state_dict[new_k] = v
        mk, uk = self.caption_model.load_state_dict(state_dict, strict=False)
        assert len(mk) == 0
        assert all([name.startswith('clip') for name in uk])
        self.caption_model.eval()
        self.caption_model.to(device)
        self.caption_model.requires_grad_(False)
        self.device = device

    def encode_prefix(self, features):
        return self.caption_model.encode_prefix(features)

    def generate_captions(self, features):  # the low dimension representation of clip feature
        """
        generate captions given features
        : param features : (tensor([B x L x D]))
        : return generated_text: (list([L]))
        """

        # generate config
        use_beam_search = True

        features = torch.split(features, 1, dim=0)
        generated_captions = []
        with torch.no_grad():
            for feature in features:
                feature = self.caption_model.decode_prefix(feature.to(self.device))  # back to the clip feature
                if use_beam_search:
                    generated_captions.append(generate_beam(self.caption_model, self.tokenizer, embed=feature)[0])
                else:
                    generated_captions.append(generate2(self.caption_model, self.tokenizer, embed=feature))
        return generated_captions
