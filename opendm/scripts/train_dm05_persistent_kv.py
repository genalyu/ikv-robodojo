"""Train ordered, preprocessed episodes. See docs/persistent_kv.md for tensor schema."""
import argparse
import shutil
import json
from pathlib import Path
import torch
from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration,DM05Config
from opendm.model.dm05.streaming_training import train_episode


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--episodes',required=True,help='Directory of trusted .pt episode lists; tensor-only prepared steps')
    p.add_argument('--checkpoint',default='/home/ubuntu/genalyu/checkpoints/DM05-MEM-Robodojo-Sim')
    p.add_argument('--output',required=True)
    p.add_argument('--dino',default='/home/ubuntu/genalyu/checkpoints/dinov2-base')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--train-vlm',action='store_true',help='Also train VLM; much higher activation/optimizer memory')
    args=p.parse_args()
    files=sorted(Path(args.episodes).glob('*.pt'))
    if not files:
        p.error('no prepared episode .pt files found')
    config=DM05Config.from_pretrained(args.checkpoint)
    config.ikv_rgb_enabled=True
    config.ikv_history_capacity=320
    config.ikv_motion_only=False
    config.ikv_dino_model_path=args.dino
    model=DM05ForConditionalGeneration.from_pretrained(args.checkpoint,config=config,torch_dtype=torch.bfloat16).cuda()
    if not args.train_vlm:
        model.model.vlm.requires_grad_(False)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr)
    def move(step):
        return {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in step.items()}
    for epoch in range(args.epochs):
        for path in files:
            # Tensor-only format; no arbitrary dataset object deserialization.
            steps=torch.load(path,map_location='cpu',weights_only=True)
            # Keep raw episode tensors on CPU; only one step enters GPU at once.
            class Steps:
                def __len__(self): return len(steps)
                def __iter__(self):
                    for step in steps: yield move(step)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=train_episode(model,Steps(),optimizer)
            print(dict(epoch=epoch,episode=path.name,loss=loss),flush=True)
        out=Path(args.output)/f'epoch-{epoch+1}'
        model.save_pretrained(out)
        for name in ('norm_stats.json','preprocessor_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','chat_template.jinja'):
            source=Path(args.checkpoint)/name
            if source.is_file():
                shutil.copy2(source,out/name)
        (out/'persistent_kv_training.json').write_text(json.dumps(dict(source_checkpoint=args.checkpoint,history_capacity=320,score='T+C+R/(1+live_semantic_repeats)',train_vlm=args.train_vlm,epoch=epoch+1),indent=2))
        torch.save(optimizer.state_dict(),out/'optimizer.pt')

if __name__=='__main__': main()
