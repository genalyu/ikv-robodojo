"""Install official pinned training environments, preferring domestic mirrors."""
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path('/mnt/cfs/9wt59p/genalyu/robodojo-posttrain')
REPO = Path('/mnt/cfs/9wt59p/genalyu/ikv-robodojo')
MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/simple'
TORCH_MIRROR = 'https://mirror.sjtu.edu.cn/pytorch-wheels/'

def start(method):
    pid_file = ROOT / f'{method}_install.pid'
    if pid_file.exists():
        try:
            old = int(pid_file.read_text())
            cmdline = Path(f'/proc/{old}/cmdline').read_bytes()
            if b'install' in cmdline:
                if os.getpgid(old) == old:
                    os.killpg(old, signal.SIGTERM)
                else:
                    os.kill(old, signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            pass
    env = dict(os.environ, UV_HTTP_TIMEOUT='300', UV_HTTP_RETRIES='5',
               UV_CACHE_DIR=str(ROOT/'uv-cache'), TMPDIR=str(ROOT/'tmp'),
               HTTP_PROXY='http://127.0.0.1:7890', HTTPS_PROXY='http://127.0.0.1:7890',
               NO_PROXY='localhost,127.0.0.1,.tsinghua.edu.cn,.sjtu.edu.cn,.aliyun.com,.modelscope.cn',
               UV_PYTHON_INSTALL_DIR='/opt/robodojo-python', DS_BUILD_OPS='0', GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.version', GIT_CONFIG_VALUE_0='HTTP/1.1')
    python = str(ROOT/'envs'/method/'bin/python')
    if method == 'openwam':
        version = subprocess.check_output([python,'-c','import sys; print(sys.version_info[:2])'],text=True).strip()
        if version != '(3, 12)':
            subprocess.run(['uv','python','install','3.12'],env=env,check=True)
            managed = subprocess.check_output(['uv','python','find','--managed-python','3.12'],env=env,text=True).strip()
            subprocess.run(['uv','venv','--clear','--python',managed,str(ROOT/'envs'/method)],env=env,check=True)
    common = ['uv', 'pip', 'install' , '--python', python, '--default-index', MIRROR,
              '--index-strategy', 'unsafe-best-match']
    if method == 'opendm':
        commands = [common + ['--index', TORCH_MIRROR+'cu128', '-e', str(REPO/'opendm'),
                             'torch==2.11.0+cu128', 'torchvision==0.26.0+cu128']]
    elif method == 'openwam':
        locked = REPO/'OpenWAM/docker/requirements-cu128.txt'
        base = ROOT/'openwam-base-requirements.txt'
        base.write_text('\n'.join(line for line in locked.read_text().splitlines()
                                  if not line.startswith('deepspeed=='))+'\n')
        commands = [
            common + ['-c',str(locked),'pip','setuptools','wheel','packaging','ninja'],
            ['uv','pip','install','--python',python,'--no-deps','--default-index',TORCH_MIRROR+'cu128',
             '-c',str(locked),'torch','torchvision','torchaudio'],
            common + ['--no-build-isolation','-r',str(base)],
            common + ['--no-build-isolation','-r',str(locked)],
            ['uv','pip','install','--python',python,'--no-deps','--no-build-isolation','-e',str(REPO/'OpenWAM')],
            ['uv','pip','install','--python',python,'--default-index','https://mirrors.aliyun.com/pypi/simple','nvidia-ml-py'],
        ]
    elif method == 'openpi':
        requirements = ROOT/'openpi-locked-requirements.txt'
        subprocess.run(['uv','export','--project',str(REPO/'openpi'),'--frozen','--no-dev',
                        '--no-emit-workspace','--no-hashes','--output-file',str(requirements)],
                       env=env,check=True,stdout=subprocess.DEVNULL)
        commands = [common + ['--index', TORCH_MIRROR+'cu126', '-r', str(requirements),
                             'torch==2.7.1+cu126', 'torchvision==0.22.1+cu126'],
                    ['uv','pip','install','--python',python,'--no-deps','-e',str(REPO/'openpi'),
                     '-e',str(REPO/'openpi/packages/openpi-client')],
                    ['uv','pip','install','--python',python,'--default-index','https://mirrors.aliyun.com/pypi/simple',
                     'accelerate==1.14.0'],
                    ['uv','pip','install','--python',python,'--no-deps','--default-index','https://mirrors.aliyun.com/pypi/simple',
                     'lerobot==0.4.4'],
                    ['uv','pip','install','--python',python,'--default-index','https://mirrors.aliyun.com/pypi/simple',
                     'datasets>=4,<5','huggingface-hub>=0.34.2,<0.36','av>=15,<16',
                     'wandb>=0.24,<0.25','gymnasium>=1.1.1,<2',
                     'pyserial>=3.5,<4','numpy==1.26.4'],
                    ['uv','pip','install','--python',python,'--no-deps','--default-index','https://mirrors.aliyun.com/pypi/simple',
                     'rerun-sdk==0.23.1']]
    else:
        raise ValueError(method)
    import shlex
    script = 'set -e\n' + '\n'.join(shlex.join(cmd) for cmd in commands)
    script_path = ROOT/f'{method}_install_domestic.sh'
    script_path.write_text(script+'\n')
    with (ROOT/f'{method}_install.log').open('a') as log:
        child = subprocess.Popen(['bash',str(script_path)],env=env,cwd=REPO,
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True)
    pid_file.write_text(str(child.pid))
    print(method,child.pid,flush=True)

if __name__ == '__main__':
    for method in sys.argv[1:] or ['opendm','openwam','openpi']:
        start(method)
