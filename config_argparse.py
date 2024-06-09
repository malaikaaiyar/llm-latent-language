import argparse
from dataclasses import fields

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def try_parse_args(cfg):
    try:
        # The get_ipython function is available in IPython environments
        ipython = get_ipython()
        if 'IPKernelApp' not in ipython.config:  # Check if not within an IPython kernel
            raise ImportError("Not in IPython")
        print("Enabling autoreload in IPython.")
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
        return cfg # use default args in notebook
    except Exception as e:
        print(f"Not in an IPython environment: {e}")
        # Parse command line arguments
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--log_file", type=str, default="experiment.log", help="File to write experiment log to")
        # cli_args = parser.parse_args()
        # print(f"Writing experiment log to {cli_args.log_file}")
        cfg = parse_args(cfg)
        #pprint.pprint(asdict(cfg))
        return cfg

def parse_args(cfg):
    parser = argparse.ArgumentParser(description="Configure model and processing parameters.")
    cls_fields = fields(cfg)

    for field in cls_fields:
        if isinstance(getattr(cfg, field.name), bool):
            # For boolean fields, use the custom str2bool function to parse them
            parser.add_argument(f"--{field.name}",
                                type=str2bool,
                                default=getattr(cfg, field.name),
                                help=f"{field.name} (default: {getattr(cfg, field.name)})")
        else:
            parser.add_argument(f"--{field.name}",
                                type=type(getattr(cfg, field.name)),
                                default=getattr(cfg, field.name),
                                help=f"{field.name} (default: {getattr(cfg, field.name)})")

    args = parser.parse_args()
    
    # Update the cfg instance with provided command line args
    args_dict = vars(args)
    for key, value in args_dict.items():
        setattr(cfg, key, value)

    # Print all parsed arguments and their values
    print("Parsed Configuration:")
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    return cfg
