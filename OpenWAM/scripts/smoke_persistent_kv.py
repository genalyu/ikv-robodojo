import time,json
from pathlib import Path
import numpy as np
from PIL import Image
from openwam.deploy.server import _load_deploy_yaml,build_server_from_config
cfg=_load_deploy_yaml('/home/ubuntu/genalyu/configs/openwam_ikv_deploy.yml')
cfg.inference.denoise_steps=2
server=build_server_from_config(cfg,'/home/ubuntu/genalyu/checkpoints/OpenWAM-Alpha-Sim-RoboDojo',device='cuda')
server._init_policy()
policy=server._policy
report=[]
for step in range(2):
    image=Image.fromarray(np.full((384,320,3),step*20,dtype=np.uint8))
    conditions=policy._build_conditions(dict(image=image,prompt='Cover the blocks.',state=np.zeros(20,dtype=np.float32)))
    start=time.time()
    result=server.engine.generate(conditions)
    bank=policy._ikv_kv_state['bank']
    row=dict(step=step,seconds=time.time()-start,action_shape=list(result['actions'].shape),finite=bool(np.isfinite(result['actions']).all()),generation=bank.generation,capacity=bank.capacity,slots=len(bank.metadata['ids']),layers=len(bank.layers),kv_shape=list(bank.layers[0][0].shape),past_slots=int((bank.metadata['ids']<step*bank.capacity).sum()))
    report.append(row);print(json.dumps(row),flush=True)
Path('/home/ubuntu/genalyu/logs/openwam_persistent_kv_smoke.json').write_text(json.dumps(report,indent=2))
