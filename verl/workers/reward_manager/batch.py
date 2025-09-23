# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )
        # print(f"DEBUG: 🏁 BatchRewardManager.verify() got scores type: {type(scores)}")
        # print(f"DEBUG: 🏁 First few scores: {scores[:2] if len(scores) > 2 else scores}")
        # if scores and isinstance(scores[0], dict):
        #     print(f"DEBUG: 🏁 First score is dict with keys: {list(scores[0].keys())}")

        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed: dict[str, Any] = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]
            # print(f"DEBUG: 🏁 Processing item {i}: score type={type(score)}, value={score}")

            if isinstance(score, dict):
                reward = score["score"]
                # print(f"DEBUG: 🏁 Item {i} is dict with keys: {list(score.keys())}")
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                    # print(f"DEBUG: 🏁   Added {key}={value} to reward_extra_info")
            else:
                print(f"DEBUG: 🏁 Item {i} is not dict, using as raw reward")
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        # print(f"DEBUG: 🏁 Final reward_extra_info keys: {list(reward_extra_info.keys())}")
        # print(f"DEBUG: 🏁 Final reward_extra_info contents:")
        # for key, values in reward_extra_info.items():
        #     print(f"DEBUG: 🏁   {key}: {values[:3] if len(values) > 3 else values} (total: {len(values)})")
        
        # Check for batch size consistency
        if reward_extra_info:
            batch_size = len(scores)
            # print(f"DEBUG: 🏁 Expected batch size: {batch_size}")
            inconsistent_keys = []
            for key, values in reward_extra_info.items():
                if len(values) != batch_size:
                    inconsistent_keys.append(f"{key}({len(values)})")
            if inconsistent_keys:
                print(f"ERROR: 🏁 Batch size mismatch for keys: {inconsistent_keys}")
                print(f"ERROR: 🏁 Full scores list: {scores}")
                # Pad missing values with 0.0 to maintain consistency
                for key, values in reward_extra_info.items():
                    while len(values) < batch_size:
                        print(f"WARNING: 🏁 Padding missing value for key '{key}' at position {len(values)}")
                        values.append(0.0)
            else:
                print(f"DEBUG: 🏁 All keys have consistent batch size")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
