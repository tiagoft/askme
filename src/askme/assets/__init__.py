import toml 

rtp_prompts = toml.load(__file__.replace('__init__.py', 'rtp_prompts.toml'))
config_data = toml.load(__file__.replace('__init__.py', 'config.toml'))