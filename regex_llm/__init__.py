from functools import partial
from typing import Any, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import regex
import torch
import torch.nn as nn
from transformers import (
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizer,
)


class AllowedTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]) -> None:
        super().__init__()
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(scores, dtype=torch.bool)
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = False
        scores = scores.masked_fill(mask, -float("inf"))
        return scores


class RegexConstraint:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: GenerationMixin,
        max_workers: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_workers = max_workers

        self._cached_vocab = self.tokenizer.get_vocab()

    def _prepare_pattern(
        self,
        pattern: Union[str, regex.Pattern[str], List[str], List[regex.Pattern[str]]],
    ) -> List[regex.Pattern[str]]:
        pattern_ = [pattern] if isinstance(pattern, (str, regex.Pattern)) else pattern
        return [regex.compile(p) for p in pattern_]

    def _check_token(
        self, current_response: str, pattern: List[regex.Pattern[str]], token: str
    ) -> bool:
        return any(p.fullmatch(current_response + token, partial=True) for p in pattern)  # type: ignore

    def _get_allowed_token_ids(
        self, current_response: str, pattern: List[regex.Pattern[str]]
    ) -> List[int]:
        with ThreadPoolExecutor() as executor:
            return [
                token_id
                for valid, token_id in zip(
                    executor.map(
                        partial(self._check_token, current_response, pattern),
                        self._cached_vocab.keys(),
                    ),
                    self._cached_vocab.values(),
                )
                if valid
            ]

    def generate(
        self,
        prompt: str,
        pattern: Union[str, regex.Pattern[str], List[str], List[regex.Pattern[str]]],
        generated_response: str = "",
        **gen_kwargs: Any,
    ) -> Optional[str]:
        pattern = self._prepare_pattern(pattern)

        input_ids: torch.Tensor = self.tokenizer.encode(prompt + generated_response, return_tensors="pt")  # type: ignore

        allowed_token_ids = self._get_allowed_token_ids(generated_response, pattern)
        # stop if no tokens are allowed
        if not allowed_token_ids:
            return None

        gen_kwargs.pop("max_new_tokens", None)
        gen_kwargs.pop("output_scores", None)
        gen_kwargs.pop("return_dict_in_generate", None)
        model_kwargs = gen_kwargs.copy()

        logits_processor = LogitsProcessorList(
            [AllowedTokenLogitsProcessor(allowed_token_ids)]
        )
        if processors := model_kwargs.pop("logits_processor", None):
            logits_processor.extend(processors)

        output = self.model.generate(
            input_ids,
            logits_processor=logits_processor,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            **model_kwargs,
        )
        probs = nn.functional.softmax(output.scores[0], dim=-1)
        sorted, indices = torch.sort(probs, descending=True, dim=-1)
        output_token_ids = indices[sorted > 0]

        for token in output_token_ids:
            output_text = self.tokenizer.decode(token, skip_special_tokens=True)
            generated_response += output_text
            print("Generated:", generated_response)

            # stop if the generated text matches the pattern
            if any(p.fullmatch(generated_response) for p in pattern):
                return generated_response

            if result := self.generate(
                prompt, pattern, generated_response, **gen_kwargs
            ):
                return result
