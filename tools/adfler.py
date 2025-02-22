import requests
import time
import os
import sys
import json

import torch
import pandas as pd
from tqdm import tqdm
from os import system, name
from datetime import datetime
from simpletransformers.ner import NERModel

from generate_report import generate_report
from parse import read_android_log, read_ios_log


def get_config():
    config_file = open('config.json')
    config_file = json.load(config_file)

    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config_file['output_dir'], now)
    # output_dir = os.path.join(config_file['output_dir'], '27112022_190057')
    previous_step = 0
    previous_status = False
    use_cuda = True if torch.cuda.is_available() == True else False
    model_config = open(os.path.join(config_file['model_dir'], 'config.json'))
    model_config = json.load(model_config)
    labels = [key for key, _ in model_config['label2id'].items()]

    wkhtml_path = ""
    if name == 'nt':
        wkhtml_path = config_file['wkhtml_path']['windows']
    # for mac and linux(here, os.name is 'posix')
    else:
        wkhtml_path = config_file['wkhtml_path']['linux']

    return {
        "output_dir": output_dir,
        "model_dir": config_file['model_dir'],
        "previous_step": previous_step,
        "previous_status": previous_status,
        "wkhtml_path": wkhtml_path,
        "app_version": config_file['app_version'],
        "use_cuda": use_cuda,
        "evidence_dir": config_file['source_evidence'],
        "labels": labels,
    }

def clear_screen():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')


def convert_time_format(time_str):
    """Detect whether time is in 12-hour or 24-hour format, and handle missing milliseconds."""
    try:
        # Try parsing 12-hour format with milliseconds
        return pd.to_datetime(time_str, format='%I:%M:%S.%f %p')
    except ValueError:
        try:
            # Try parsing 12-hour format without milliseconds
            return pd.to_datetime(time_str, format='%I:%M:%S %p')
        except ValueError:
            try:
                # Try parsing 24-hour format with milliseconds
                return pd.to_datetime(time_str, format='%H:%M:%S.%f')
            except ValueError:
                # Try parsing 24-hour format without milliseconds
                return pd.to_datetime(time_str, format='%H:%M:%S')


def menu():
    clear_screen()
    print("\t\t====================================================================")
    print("\t\t==============   Drone Flight Log Event Recognizer   ==============")
    print("\t\t====================================================================\n")
    print("\t\tAction to perform:\n")
    print("\t\t\t1. Evidence Checking")
    print("\t\t\t2. Forensic Timeline Construction")
    print("\t\t\t3. Drone Event Recognition")
    print("\t\t\t4. Forensic Report Generation")
    print("\t\t\t0. Exit\n")
    try:
        option = input("\t\tEnter option: ")
    except EOFError:
        option = "1"
    return option


def main():
    # now = datetime.now()
    # now = now.strftime("%d%m%Y_%H%M%S")
    # output_dir = os.path.join("./result", now)
    config = get_config()

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    start = menu()
    if start == '0':
        with open(config['output_dir'] + '/config.json', 'w') as file:
                json.dump(config, file)
        print("Exit program...")
        time.sleep(2)
        sys.exit(0)
    while start != '0':
        if start == '0':
            with open(config['output_dir'] + '/config.json', 'w') as file:
                json.dump(config, file)
            print("Exit program...")
            time.sleep(1)
            sys.exit(0)
        elif start == '1':
            clear_screen()
            print('Evidence checking in process...\n')
            time.sleep(1)
            config['previous_step'] = 1
            
            files = os.listdir(config['evidence_dir'])
            android_logs = []
            ios_logs = []
            folders = [d for d in files if os.path.isdir(config['evidence_dir']+'/'+d)]
            # print(folders)
            if(len(folders) == 0):
                print("No sub-folders in the evidence folder")
                config['previous_status'] = False
                time.sleep(1)
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
            else: 
                for folder in folders:
                    # Filtering only the files.
                    files = os.listdir(config['evidence_dir']+'/'+folder)
                    files = [f for f in files if os.path.isfile(config['evidence_dir']+'/'+folder+'/'+f)]
                    if(folder == 'android'):
                        android_logs.append(files)
                    else:
                        ios_logs.append(files)
            android_logs.extend(ios_logs)
            # save to .json file
            
            if(len(android_logs) == 0):
                print('No found files in the evidence folder!')
                config['previous_status'] = False
                time.sleep(1)
            else:
                with open(config['output_dir'] + '/raw_list.json', 'w') as file:
                    json.dump(android_logs, file)
                config['previous_status'] = True
                time.sleep(1)
                print('Found files: \n')
                print('iOS logs: ')
                print(*ios_logs, sep="\n")
                print("\nAndroid logs: \n")
                print(*android_logs, sep="\n")
                print('Finish checking evidence...')
                time.sleep(1)
            try:
                input("Press enter to continue...")
            except EOFError:
                print("No input received, exit program...")
                sys.exit(0)
        elif start == '2':
            if config['previous_status'] == False and config['previous_step'] == 1:
                print('Previous step is not complete, please return to previous step')
                time.sleep(1)
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
            elif (config['previous_step'] == 1 and config['previous_status'] == True) or (config['previous_step'] != 1 and config['previous_status'] == True):
                clear_screen()
                print('Forensic timeline construction is in process...\n')
                config['previous_step'] = 2
                # Parse the raw flight logs
                os.makedirs(config['output_dir'] + '/parsed/android')
                android_path = os.path.join(config['output_dir'], 'parsed/android')
                os.makedirs(config['output_dir'] + '/parsed/ios')
                ios_path = os.path.join(config['output_dir'], 'parsed/ios')
                full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['evidence_dir'])
                
                # Construct the forensic timeline from parsed flight log
                # print(full_path)
                # print(os.path.join(dir_path, config['evidence_dir']))
                path_list = []
                ios_parsed = False
                android_parsed = False
                for path, subdirs, files in os.walk(full_path):
                    if path.find("android") != -1:
                        for filename in os.listdir(path):
                            if filename.find("parsed") != -1:
                                continue
                            print("path: ", path)
                            print("Extracting file: %s" % filename)
                            read_android_log(path, filename, android_path)
                            print("Finish Extracting file: %s\n" % filename)
                        android_parsed = True
                        
                    elif path.find("ios") != -1:
                        for filename in os.listdir(path):
                            if filename.find("parsed") != -1:
                                continue
                            print("path: ", path)
                            print("Extracting file: %s" % filename)
                            read_ios_log(path, filename, ios_path)
                            print("Finish Extracting file: %s\n" % filename)
                        ios_parsed = True

                parsed_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join(config['output_dir'], 'parsed'))
                for path, subdirs, files in os.walk(parsed_path):
                    for filename in files:
                        path_list.append(os.path.join(path, filename))
                    # if(ios_parsed or android_parsed):
                    #     for name in files:
                    #         file_ext = name.split(".")
                    #         file_ext = file_ext[-1] if len(file_ext) > 1 else ""
                    #         if(name.find("parsed_") != -1 and file_ext == "csv"):
                    #             path_list.append(os.path.join(path, name))

                parent_df = pd.DataFrame()
                if(len(path_list) == 0):
                    print('No parsed evidence found.')
                    config['previous_status'] = False
                    time.sleep(1)
                    try:
                        input("Press enter to continue...")
                    except EOFError:
                        print("No input received, exit program...")
                        sys.exit(0)

                for path in path_list:
                    child_df = pd.read_excel(path)
                    parent_df = pd.concat([parent_df, child_df])
                # format the time properly
                parent_df['time'] = parent_df['time'].apply(convert_time_format).dt.time
                parent_df['time'] = parent_df['time'].astype(str)
                parent_df['date'] = parent_df['date'].astype(str)
                parent_df['timestamp'] =  parent_df[['time', 'date']].agg(' '.join, axis=1)

                parent_df.drop(columns = ['time', 'date'], inplace=True)
                parent_df = parent_df[['timestamp', 'message']]
                parent_df['timestamp'] = pd.to_datetime(parent_df['timestamp'])
                # sort the data by timestamp
                parent_df.sort_values(by='timestamp', inplace=True)

                print('Save forensic timeline to .xlsx file...')
                parent_df.to_excel(config['output_dir'] + '/forensic_timeline.xlsx', index=False)             
                
                print('Finish constructing timeline.')
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
            else:
                print('Please follow the steps accordingly')
                time.sleep(1)
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
        elif start == '3':
            if config['previous_status'] == False and config['previous_step'] == 2:
                print('Previous step is not complete, please return to previous step')
                time.sleep(1)
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
            elif (config['previous_step'] == 2 and config['previous_status'] == True) or (config['previous_step'] != 2 and config['previous_status'] == True):
                clear_screen()
                print('Event Recognition is in process...\n')
                config['previous_step'] == 3
                # Load the fine-tuned model
                print("Loading model...\n")
                model_exist = os.path.exists(config['model_dir'] + '/pytorch_model.bin')
                if (model_exist == False):
                    print('The model file is not found.')
                    config['previous_status'] = False
                    time.sleep(1)
                    try:
                        input("Press enter to continue...")
                    except EOFError:
                        print("No input received, exit program...")
                        sys.exit(0)
                else:
                    print(config['model_dir'])
                    droner = NERModel(
                        "bert", 
                        config['model_dir'],
                        use_cuda=config['use_cuda']
                    )
                    print("Model is loaded successfully\n")
                    # Load the forensic timeline
                    print("Loading forensic timeline...\n")
                    timeline_exist = os.path.exists(config['output_dir'] + '/forensic_timeline.xlsx')
                    if(timeline_exist == False): 
                        print('The forensic timeline file is not found.')
                        config['previous_status'] = False
                        time.sleep(1)
                        try:
                            input("Press enter to continue...")
                        except EOFError:
                            print("No input received, exit program...")
                            sys.exit(0)
                    else:
                        timeline = pd.read_excel(config['output_dir'] + '/forensic_timeline.xlsx')
                        print("Forensic timeline is loaded successfully\n")
                        print('Start recognizing mentioned entities...')
                        pred_list = []
                        for row in tqdm(range(0, timeline.shape[0])):
                            message = timeline.iloc[row, 1]
                            [entities], _ = droner.predict([message])
                            timestamp = timeline.iloc[row, 0]
                            pred_list.append({"timestamp": str(timestamp), "entities": entities})
                        
                        # save to .json file
                        with open(config['output_dir'] + '/ner_result.json', 'w') as file:
                            json.dump(pred_list, file)
                        print('Finish recognizing mentioned entities...')
                        time.sleep(1)
                        try:
                            input("Press enter to continue...")
                        except EOFError:
                            print("No input received, exit program...")
                            sys.exit(0)
            else:
                print('Please follow the steps accordingly')
                time.sleep(1)
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
        elif start == '4':
            if config['previous_status'] == False and config['previous_step'] == 3:
                print('Previous step is not complete, please return to previous step')
                time.sleep(1)
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
            elif (config['previous_step'] == 2 and config['previous_status'] == True) or (config['previous_step'] != 2 and config['previous_status'] == True):
                clear_screen()
                print('Forensic report generation is in process...\n')
                config['previous_step'] == 4
                print('Loading the NER results...')
                # Opening JSON file
                ner_result_exist = os.path.exists(config['output_dir'] + '/ner_result.json')
                if(ner_result_exist == False):
                    print('The NER result is not found.')
                    config['previous_status'] = False
                    time.sleep(1)
                    try:
                        input("Press enter to continue...")
                    except EOFError:
                        print("No input received, exit program...")
                        sys.exit(0)
                else:
                    # Load the NER results
                    timeline_file = open(config['output_dir'] +  '/ner_result.json')
                    timeline = json.load(timeline_file)
                    print('NER result is loaded successfully.')

                    print('Start generating forensic report...')
                    try:
                        generate_report(config)
                    except:
                        print('Error in generating report.')
                        config['previous_status'] = False
                        time.sleep(1)
                        try:
                            input("Press enter to continue...")
                        except EOFError:
                            print("No input received, exit program...")
                            sys.exit(0)
                    else:
                        print('Report has generated successfully.')
                        try:
                            input("Press enter to continue...")
                        except EOFError:
                            print("No input received, exit program...")
                            sys.exit(0)
            else:
                print('Please follow the steps accordingly')
                time.sleep(1)
                try:
                    input("Press enter to continue...")
                except EOFError:
                    print("No input received, exit program...")
                    sys.exit(0)
        else:
            print('Invalid option!')
            try:
                input("Press enter to continue...")
            except EOFError:
                print("No input received, exit program...")
                sys.exit(0)
        start = menu()
    sys.exit(0)


if __name__ == "__main__":
    main()