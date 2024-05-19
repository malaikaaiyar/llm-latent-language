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

def parse_args(config_class):
    parser = argparse.ArgumentParser(description="Configure model and processing parameters.")
    cls_fields = fields(config_class)

    for field in cls_fields:
        if isinstance(getattr(config_class, field.name), bool):
            # For boolean fields, use the custom str2bool function to parse them
            parser.add_argument(f"--{field.name}",
                                type=str2bool,
                                default=getattr(config_class, field.name),
                                help=f"{field.name} (default: {getattr(config_class, field.name)})")
        else:
            parser.add_argument(f"--{field.name}",
                                type=type(getattr(config_class, field.name)),
                                default=getattr(config_class, field.name),
                                help=f"{field.name} (default: {getattr(config_class, field.name)})")

    args = parser.parse_args()
    
    # Update the config_class instance with provided command line args
    args_dict = vars(args)
    for key, value in args_dict.items():
        setattr(config_class, key, value)

    # Print all parsed arguments and their values
    print("Parsed Configuration:")
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    return config_class
