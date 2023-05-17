# regex llm

Constrained language model generation with regex.

## Installation

```bash
pip install regex-llm
```

## Usage

```python
import regex
from transformers import (
    AutoTokenizer,
    GenerationMixin,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
)

from regex_llm import RegexConstraint

model: GenerationMixin = AutoModelForCausalLM.from_pretrained("your model")
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("your model")

controller = RegexConstraint(tokenizer=tokenizer, model=model)

prompt = '帮我生成3种不同的水果\n输出json格式如下：\n[{"name":"xxx","description":"xxx"}]'
pattern = r"\[(,?\s*\{\"name\":\s*\"[^\"]+\",\s*\"description\":\s*\"[^\"]+\"\\s*})+\s*\]"
output = controller.generate(
    prompt=prompt,
    pattern=pattern,
    do_sample=True,
    num_beams=1,
    top_p=0.7,
    temperature=0.95,
)
print(output)
```
