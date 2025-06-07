# utils/debug.py

import os
import json

def setup_debug(is_main_process, port=10099):
    # assert debugpy is installed
    try:
        import debugpy
    except ImportError:
        raise ImportError("`debugpy` is not installed. Please install it if you call 'setup_debug' .")
    
    master_addr = os.environ['SLURM_NODELIST'].split(',')[0]
    
    # Auto-configure .vscode/launch.json
    launch_json_path = os.path.join('.vscode', 'launch.json')
    os.makedirs(os.path.dirname(launch_json_path), exist_ok=True)
    
    default_config = {
        "version": "0.2.0",
        "configurations": [{
            "name": "slurm_debug",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": master_addr,
                "port": port
            },
            "justMyCode": True
        }]
    }
    
    # config updater
    try:
        with open(launch_json_path, 'r') as f:
            existing_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_config = default_config
    
    if existing_config.get('configurations'):
        existing_config['configurations'][0]['connect'].update({
            "port": port,
            "host": master_addr
        })
    
    with open(launch_json_path, 'w') as f:
        json.dump(existing_config, f, indent=4)

    if is_main_process:  # 🎯 Master process handler
        print(f"🚨 Debug portal active on {master_addr}:{port}", flush=True)
        debugpy.listen((master_addr, port))
        debugpy.wait_for_client()
        print("🔗 Debugger linked!", flush=True)