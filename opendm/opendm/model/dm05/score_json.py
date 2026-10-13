"""Constrain JSON structure and IDs; all 0..100 importance values remain model choices."""
import json
class ScoreJSONConstraint:
    def __init__(self, tokenizer, candidates, prompt_length):
        self.prompt_length=prompt_length
        self.tokenizer=tokenizer
        self.nodes=[{}]
        self.cursor=0
        self.consumed=0
        stages=[]
        def fixed(text):
            stages.append(("fixed",tokenizer.encode(text,add_special_tokens=False)))
        fixed('{"visual_scores":[')
        for modality in ("visual","tactile"):
            if modality=="tactile":fixed('],"tactile_scores":[')
            rows=[c for c in candidates if c["modality"]==modality]
            for i,c in enumerate(rows):
                fixed(("," if i else "")+'{"id":'+json.dumps(c["id"])+',"score":')
                stages.append(("number",[tokenizer.encode(str(n),add_special_tokens=False) for n in range(101)]))
                fixed("}")
        fixed("]}")
        def node(edges=None):
            self.nodes.append(dict(edges or {}));return len(self.nodes)-1
        end=node()
        eos=tokenizer.eos_token_id
        if eos is None:raise ValueError("Tokenizer has no EOS")
        self.nodes[end][eos]=end
        next_node=end
        self.max_tokens=1
        for kind,data in reversed(stages):
            if kind=="fixed":
                self.max_tokens+=len(data)
                for token in reversed(data):next_node=node({token:next_node})
            else:
                self.max_tokens+=max(map(len,data))
                root=node()
                for path in data:
                    current=root
                    for token in path:
                        if token not in self.nodes[current]:self.nodes[current][token]=node()
                        current=self.nodes[current][token]
                    for token,target in self.nodes[next_node].items():
                        prior=self.nodes[current].get(token)
                        if prior is not None and prior!=target:raise ValueError("Ambiguous number encoding")
                        self.nodes[current][token]=target
                next_node=root
        self.cursor=next_node
    def __call__(self,batch_id,input_ids):
        if batch_id!=0:raise ValueError("Scoring grammar supports one request")
        generated=input_ids[self.prompt_length+self.consumed:].tolist()
        for token in generated:
            self.cursor=self.nodes[self.cursor][token]
            self.consumed+=1
        return list(self.nodes[self.cursor])
