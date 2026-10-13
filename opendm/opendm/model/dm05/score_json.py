"""Exact score JSON grammar at character level, preserving every valid BPE segmentation."""
import json
import transformers.tokenization_utils as _tokenization
from transformers import PreTrainedTokenizerBase
if not hasattr(_tokenization, "PreTrainedTokenizerBase"):
    _tokenization.PreTrainedTokenizerBase=PreTrainedTokenizerBase
from lmformatenforcer.characterlevelparser import CharacterLevelParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
_DATA={}
_NUMBERS=tuple(str(i) for i in range(101))
_SPACE=" \n\r\t"

class ExactScoreParser(CharacterLevelParser):
    def __init__(self, stages, index=0, position=0, number="", spaces=0, config=None):
        super().__init__(config)
        self.stages=stages;self.index=index;self.position=position
        self.number=number;self.spaces=spaces
    def state(self,index=None,position=0,number="",spaces=0):
        return ExactScoreParser(self.stages,self.index if index is None else index,
                                position,number,spaces,self.config)
    def next(self):
        return self.state(self.index+1)
    def can_end(self):
        if self.index==len(self.stages):return True
        return self.stages[self.index][0]=="space" and self.next().can_end()
    def get_allowed_characters(self):
        if self.index==len(self.stages):return ""
        kind,value=self.stages[self.index]
        if kind=="text":return value[self.position]
        if kind=="space":
            return (_SPACE if self.spaces<1 else "")+self.next().get_allowed_characters()
        digits={n[len(self.number)] for n in _NUMBERS
                if n.startswith(self.number) and len(n)>len(self.number)}
        return "".join(sorted(digits))+(self.next().get_allowed_characters() if self.number in _NUMBERS else "")
    def add_character(self,char):
        if char not in self.get_allowed_characters():raise ValueError("Invalid score JSON character")
        kind,value=self.stages[self.index]
        if kind=="text":
            if self.position+1==len(value):return self.next()
            return self.state(position=self.position+1)
        if kind=="space":
            if char in _SPACE and self.spaces<1:return self.state(spaces=self.spaces+1)
            return self.next().add_character(char)
        if any(n.startswith(self.number+char) for n in _NUMBERS):
            return self.state(number=self.number+char)
        return self.next().add_character(char)
    def cache_key(self):
        return self.index,self.position,self.number,self.spaces

class ScoreJSONConstraint:
    def __init__(self, tokenizer, candidates, prompt_length):
        stages=[]
        def literal(value):
            if value:stages.append(("text",value))
        def space():stages.append(("space",None))
        def atom(value):literal(value);space()
        atom("{");atom('"visual_scores"');atom(":");atom("[")
        for modality in ("visual","tactile"):
            if modality=="tactile":
                atom("]");atom(",");atom('"tactile_scores"');atom(":");atom("[")
            for i,c in enumerate(x for x in candidates if x["modality"]==modality):
                if i:atom(",")
                atom("{");atom('"id"');atom(":");atom(json.dumps(c["id"]))
                atom(",");atom('"score"');atom(":")
                stages.append(("score",None));space();atom("}")
        atom("]");atom("}")
        self.stages=tuple(stages)
        key=id(tokenizer)
        if key not in _DATA:_DATA[key]=build_token_enforcer_tokenizer_data(tokenizer)
        self.function=build_transformers_prefix_allowed_tokens_fn(_DATA[key],ExactScoreParser(self.stages))
        self.max_tokens=sum(len(value) if kind=="text" else 1 if kind=="space" else 3 for kind,value in stages)+1
    def __call__(self,batch_id,input_ids):
        return self.function(batch_id,input_ids)

