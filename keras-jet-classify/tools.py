import yaml
import csv

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)