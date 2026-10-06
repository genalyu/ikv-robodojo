import sys,json,time
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/genalyu/RoboDojo')
import numpy as np
import torch,yaml
from PIL import Image
from collections import deque
from XPolicyLab.policy.OpenDM.model import Model
cfg=yaml.safe_load(Path('/home/ubuntu/genalyu/configs/opendm_ikv.yml').read_text())
cfg.update(model_path='/home/ubuntu/genalyu/checkpoints/DM05-MEM-Robodojo-Sim',norm_stats_path='/home/ubuntu/genalyu/checkpoints/DM05-MEM-Robodojo-Sim/norm_stats.json',env_cfg_type='arx_x5',diffusion_steps=2)
m=Model(cfg)
rgb=np.zeros((256,256,3),dtype=np.uint8)
image=Image.fromarray(rgb)
report=[]
for step in [500,525]:
    m._history_by_env[0]=deque([(i+1,i,rgb) for i in range(step-500,step+1,25)],maxlen=21)
    m._history_sequence_by_env[0]=step+1
    start=time.time()
    history=m._online_history_payload_for(0)
    payload=dict(images=[image]*3,history_images=[],**history,prompt='Cover the blocks.',state=np.zeros(14,dtype=np.float32),meta_data=dict(robot_type=m._robot_type,control_mode=cfg.get('control_mode'),speed='0.5',state_desc=m._state_desc,valid_dim_mask=np.ones(14,dtype=bool)))
    actions=m._inference._predict(payload)
    bank=m._ikv_memory_by_env[0]['bank']
    row=dict(step=step,seconds=time.time()-start,action_shape=list(actions.shape),finite=bool(np.isfinite(actions).all()),generation=bank.generation,slots=len(bank.metadata['ids']),layers=len(bank.layers),kv_shape=list(bank.layers[0][0].shape))
    report.append(row);print(json.dumps(row),flush=True)
Path('/home/ubuntu/genalyu/logs/dm05_persistent_kv_smoke.json').write_text(json.dumps(report,indent=2))
