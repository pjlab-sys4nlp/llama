import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from sentencepiece import SentencePieceProcessor
from tqdm import trange
import fire
import json

import io, os, sys

from llama import ModelArgs, Transformer, myLLaMA

backup = {}

def proxy_off():
    global backup
    if 'http_proxy' in os.environ:
        backup['http_proxy'] = os.environ['http_proxy']
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        backup['https_proxy'] = os.environ['https_proxy']
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        backup['HTTP_PROXY'] = os.environ['HTTP_PROXY']
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        backup['HTTPS_PROXY'] = os.environ['HTTPS_PROXY']
        del os.environ['HTTPS_PROXY']

def proxy_on():
    global backup
    if 'http_proxy' in backup:
        os.environ['http_proxy'] = backup['http_proxy']
    if 'https_proxy' in backup:
        os.environ['https_proxy'] = backup['https_proxy']
    if 'HTTP_PROXY' in backup:
        os.environ['HTTP_PROXY'] = backup['HTTP_PROXY']
    if 'HTTPS_PROXY' in backup:
        os.environ['HTTPS_PROXY'] = backup['HTTPS_PROXY']

def get_parallel_conf():
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1)) 
    return local_rank, world_size

def setup_model_parallel(init_seed=1):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    torch.distributed.init_process_group('nccl')
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(init_seed)
    return local_rank, world_size

def load_all(ckpt_dir, tokenizer_path, local_rank, max_seq_len, max_batch_size):
    proxy_off()
    
    if ckpt_dir.startswith('s3://'):
        assert ckpt_dir.startswith('s3://model_weights/0331/evaluation/exported_llama/'), 'Read the code!'
        try:
            from petrel_client.client import Client
        except ImportError as e:
            print('model weight on s3 require aws SDK, see http://sdoc.pjlab.org.cn:10099/docs/')
            exit()
        client = Client()
    else:
        try:
            ckpt_dir = os.path.abspath(ckpt_dir)
        except:
            print('I don\'t know how to get here, check your ckpt_dir.')
            exit()
        assert ckpt_dir.startswith('/'), 'Read the code!'
        # replace with your local model weight folder
        # assert ckpt_dir.startswith('/mnt/petrelfs/share_data/llm_weight/evaluation/exported_llama/'), 'Read the code!'
        client = None
    
    print(f'{tokenizer_path} is loading.', flush=True)
    tokenizer = SentencePieceProcessor()
    try:
        tokenizer_path = os.path.abspath(tokenizer_path)
    except:
        print('tokenizer_path must be a local path.')
        exit()
    assert tokenizer_path.startswith('/'), 'tokenizer_path must be a local path.'
    tokenizer.load(tokenizer_path)
    print('Tokenizer loaded.')
    
    print('Config loading', flush=True)
    config_file = os.path.join(ckpt_dir, 'params.json')
    if ckpt_dir.startswith('s3://model_weights/0331/evaluation/exported_llama/'):
        assert client.contains(config_file), 'Need config file!'
        with io.BytesIO(client.get(config_file)) as f:      
            params = json.load(io.TextIOWrapper(f, encoding='utf-8'))
    elif ckpt_dir.startswith('/mnt/petrelfs/share_data/llm_weight/evaluation/exported_llama/'):
        assert os.path.exists(config_file), 'Need config file!'
        with open(config_file, 'r') as f:
            params = json.load(f)
    model_config = {'hidden_size':params['dim'], 'num_layers':params['n_layers'], 'num_attention_heads':params['n_heads']}
    print('Config done!', flush=True)
    
    model_args = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,
        dim=model_config['hidden_size'], n_layers=model_config['num_layers'],
        n_heads=model_config['num_attention_heads'], vocab_size=tokenizer.vocab_size()
    )
    
    print('model weight loading', flush=True)
    
    if ckpt_dir.startswith('s3://'):
        with io.BytesIO(client.get(f'{ckpt_dir}tp_{local_rank}.pt')) as f:
            current_states = torch.load(f, map_location='cpu')
    else:
        current_states = torch.load(f'{ckpt_dir}tp_{local_rank}.pt', map_location='cpu')
        
    print('model weight loaded', flush=True)
    
    proxy_on()
    local_rank, world_size = setup_model_parallel()
    
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    
    load_info = model.load_state_dict(current_states, strict=False)
    print(load_info)
    
    return model, tokenizer
    
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    tokenizer_type: str,
    max_seq_len: int = 2048,
    max_batch_size: int = 32
):
    local_rank, world_size = get_parallel_conf()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    model, tokenizer = load_all(
        ckpt_dir, tokenizer_path, local_rank, max_seq_len, max_batch_size
    )
    
    generator = myLLaMA(model, tokenizer, tokenizer_type)
    
    batch_size = 16
    max_gen_length = 200
    input_prompts = [
        'I remember that night, I just might regret that night for the rest of my days',
        'There’s a million things I haven’t done but',
        '长太息以掩涕兮 哀民生之多艰',
        '周杰伦是华语乐坛最'
    ]
    
    with open('result.txt', 'w', encoding='utf-8') as f:
        for i in trange(0, len(input_prompts), batch_size, disable=local_rank>0):
            # temperature=0 -> greedy decode
            # else -> sampling
            results = generator.generate(input_prompts[i:i+batch_size], max_gen_len=max_gen_length, temperature=0)
            for result in results:
                if local_rank == 0:
                    f.write(f'----------------\n{result}\n----------------\n')
    
if __name__ == '__main__':
    fire.Fire(main)
