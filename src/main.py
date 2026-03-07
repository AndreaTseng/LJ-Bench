import pandas as pd
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from query_model import query_model, get_api_keys_dict
import glob
import copy


def process_attack(config_args, attack, df, client_type, run_index):
    model_name = config_args.target_model
    results = []
    
    if attack == "multi_lan":
        import multi_lan
        return multi_lan.process_multi_lan_attack(config_args, df, run_index)

    api_keys = get_api_keys_dict(config_args)
    
    if attack == "past_tense" and hasattr(config_args, 'past_tense_data') and config_args.past_tense_data:
        try:
            print(f"Past_tense dataset: {config_args.past_tense_data}")
            df = pd.read_csv(config_args.past_tense_data)
    
        except Exception as e:
            print(f"Cannot load past_tense data: {e}")


    def process_row(row):
        prompt = row['Prompt']
        response = query_model(
            prompt, 
            config_args.target_model, 
            config_args.max_output_tokens, 
            config_args.temperature, 
            attack, 
            client_type,
            api_keys,
            config_args.reasoning_effort
        )
        
        result = row.to_dict()
        result['Response'] = response
        
        return result

    max_workers = config_args.batch_size if hasattr(config_args, 'batch_size') else 5
    total_rows = len(df)
    
    if hasattr(config_args, 'output_file') and config_args.output_file:
        output_csv_file = config_args.output_file
    else:
        output_csv_file = os.path.join(config_args.output_dir, f"{model_name}_{attack}_{run_index + 1}.csv")
    
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    
    rows_processed = 0
    save_interval = 5  # save after every 5 rows

    def save_current_results():
        current_results_df = pd.DataFrame(results)
        current_results_df.to_csv(output_csv_file, index=False)
        print(f"Interim results saved: {len(results)}/{total_rows} rows processed")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start_idx in range(0, total_rows, config_args.batch_size):
            end_idx = min(start_idx + config_args.batch_size, total_rows)
            batch_rows = df.iloc[start_idx:end_idx]
            futures = [executor.submit(process_row, row) for index, row in batch_rows.iterrows()]           
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing attack: {attack}, Run: {run_index + 1}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in processing future: {e}")
                    error_result = {col: 'Error' for col in df.columns}
                    error_result['Response'] = 'Error'
                    results.append(error_result)
              
                rows_processed += 1
                if rows_processed % save_interval == 0:
                    save_current_results()
            
            time.sleep(config_args.rate_limit if hasattr(config_args, 'rate_limit') else 1)
   
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_file, index=False)
    print(f"Responses saved to {output_csv_file}")    
    return results_df

def main(config_args):

    print("\n=== Configuration Parameters ===")
    for key, value in vars(config_args).items():
        if key.endswith("API_KEY") and value:
            print(f"{key}: {value[:5]}...{value[-3:] if len(value) > 8 else ''}")
        else:
            print(f"{key}: {value}")
    print("===============================\n")
    

    if os.path.isdir(config_args.input_file):
        csv_files = glob.glob(os.path.join(config_args.input_file, "*.csv"))       
        if not csv_files:
            raise ValueError(f"No CSV files found in folder: {config_args.input_file}")
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            process_single_file(config_args, file_path, file_name)
    else:
        file_name = os.path.basename(config_args.input_file)
        process_single_file(config_args, config_args.input_file, file_name)


def process_single_file(config_args, file_path, file_name):
    print(f"Processing file: {file_path}")
    
    df = pd.read_csv(file_path)
    #required_columns = {'Type', 'Prompt', 'Category'}
    required_columns = {'Prompt'}
    if not required_columns.issubset(df.columns):
        print(f"File {file_path} does not contain required columns: {required_columns}")
        return

    client_type = config_args.target_model_type
    base_name = os.path.splitext(file_name)[0]
    print("base_name", base_name)

    if not os.path.exists(config_args.output_dir):
        os.makedirs(config_args.output_dir)
    if hasattr(config_args, 'data_folder') and config_args.data_folder:
        os.makedirs(config_args.data_folder, exist_ok=True)

    attack_types = ["baseline", "comb1", "comb2", "comb3", "DAN", "past_tense"]

    if config_args.attack.lower() == "all":
        for attack in attack_types:
            for run_index in range(config_args.start_index,  config_args.start_index + config_args.runs):
                output_file = f"{base_name}_{config_args.target_model}_{attack}_{run_index}.csv"
                output_path = os.path.join(config_args.output_dir, output_file)
                current_config = copy.deepcopy(config_args)
                current_config.attack = attack
                current_config.output_file = output_path
                
                process_attack(current_config, attack, df, client_type, run_index)
    else:
        for run_index in range(config_args.runs):
            output_file = f"{base_name}_{config_args.target_model}_{config_args.attack}_{(run_index + 1)}.csv"
            print("output filename: ", output_file)
            output_path = os.path.join(config_args.output_dir, output_file)
            current_config = copy.deepcopy(config_args)
            current_config.output_file = output_path
            
            process_attack(current_config, config_args.attack, df, client_type, run_index)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)

    if "data_folder" not in config:
        config["data_folder"] = "data"
    if "OPENAI_API_KEY" not in config and "api_key" in config:
        config["OPENAI_API_KEY"] = config["api_key"]    
    if "TOGETHER_API_KEY" not in config and "api_key" in config:
        config["TOGETHER_API_KEY"] = config["api_key"]
    if "GEMINI_API_KEY" not in config and "api_key" in config:
        config["GEMINI_API_KEY"] = config["api_key"]
    if "evaluator_model" not in config:
        config["evaluator_model"] = "gemini-1.5-pro"
    if "past_tense_data" not in config:
        config["past_tense_data"] = ""
        
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please include config file")    
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()   
    try:
        config = load_config(args.config)
        
        class ConfigNamespace:
            def __init__(self, config_dict):
                self.__dict__.update(config_dict)
        config_args = ConfigNamespace(config)
        main(config_args)
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()
