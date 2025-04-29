import numpy as np
import pandas as pd
import os
import csv
import shutil
import json

import sys 

raw_path = "../../benchmarks/anomaly/data"
base_path = ".."
export_path_uts = "datasets-main/UTS"

def convert_to_unixtime_with_offset(time, offset):
    unix_time = int(time.timestamp())
    unix_time_with_offset = unix_time + offset
    return unix_time_with_offset

# Complementing values on missing timestamps. Results saved in new_path
# Use when timestamps are integers instead of datetime

# def insert_timestamp(path, new_path, timestamp_label):
#
#     times = pd.read_csv(path, usecols=[timestamp_label])
#     times = sum(times.values.tolist(), [])
#
#
#     min_value = np.min(times)
#     times = times - min_value
#     min_interval = np.min(np.diff(times))
#
#     times = times / min_interval
#     # check all values are integer
#     times_check = times % 1
#     assert np.max(times_check) == 0
#     assert np.min(times_check) == 0
#
#     max_time = int(np.max(times))
#     if len(times) == max_time + 1:
#         print("Skip timestamp complement.")
#         shutil.copy(path, new_path)
#         return min_interval
#
#     print("max value is %d\n" % max_time)
#     with open(path, 'r') as rf:
#         rawdata = rf.readlines()
#
#     col_num = rawdata[0].count(",")
#     newdata = []
#     newdata.append(rawdata[0])
#
#     index = 0
#     for i in range(max_time + 1):
#         if times[index] == i:
#             newdata.append(rawdata[int(index + 1)])
#             index += 1
#         else:
#             newline = "%d" % (min_value + i * min_interval) + col_num * "," + '\n'
#             newdata.append(newline)
#             print("Add new line: %s" % newline[:-1])
#
#     with open(new_path, 'w', newline='') as wf:
#         wf.writelines(newdata)
#
#     return min_interval

def insert_timestamp(path, new_path, timestamp_label):
    # Attempt to detect datetime format
    try:
        # Try reading as ISO datetime
        df = pd.read_csv(path, parse_dates=[timestamp_label])
        times = df[timestamp_label].values.astype(np.int64) // 10 ** 9  # Convert to Unix timestamps
        is_datetime = True
        original_format = df[timestamp_label].dt.strftime('%Y-%m-%d %H:%M:%S').iloc[0]
    except:
        # Fallback to numeric timestamp handling
        df = pd.read_csv(path, usecols=[timestamp_label])
        times = sum(df.values.tolist(), [])  # Original flattening method
        is_datetime = False

    # Common processing for both types
    times = np.array(times)
    min_value = np.min(times)
    times_normalized = times - min_value
    min_interval = np.min(np.diff(np.unique(times_normalized)))

    # Handle zero intervals (common in test datasets)
    if min_interval == 0:
        min_interval = 1 if is_datetime else np.min(np.diff(np.unique(times_normalized[times_normalized != 0])))

    times_scaled = times_normalized / min_interval
    times_rounded = np.round(times_scaled).astype(int)

    # Check integer alignment with tolerance
    if not np.allclose(times_scaled, times_rounded, atol=1e-6):
        raise ValueError("Timestamps are not multiples of min_interval")

    max_time = int(np.max(times_rounded))
    if len(times) == max_time + 1:
        print("Skip timestamp complement.")
        shutil.copy(path, new_path)
        return min_interval

    # Read original file for reconstruction
    with open(path, 'r') as rf:
        rawdata = rf.readlines()

    col_num = rawdata[0].count(",")
    newdata = [rawdata[0]]
    index = 0

    # Rebuild timeline
    for i in range(max_time + 1):
        if index < len(times_rounded) and times_rounded[index] == i:
            newdata.append(rawdata[index + 1])
            index += 1
        else:
            if is_datetime:
                # Convert back to datetime string
                unix_ts = min_value + i * min_interval
                dt = pd.to_datetime(unix_ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')
                newline = f"{dt}{',' * col_num}\n"
            else:
                # Original numeric handling
                new_ts = min_value + i * min_interval
                newline = f"{new_ts}{',' * col_num}\n"

            newdata.append(newline)
            print(f"Added missing timestamp: {newline.strip()}")

    # Write complemented file
    with open(new_path, 'w', newline='') as wf:
        wf.writelines(newdata)

    return min_interval


# fill n/a
def fillna(path, usecols):
    data = pd.read_csv(path, usecols=usecols)
    # data.interpolate(method='linear', inplace=True)
    # data.fillna(method='ffill', inplace=True)
    # data.fillna(method='bfill', inplace=True)
    # Convert to numeric where needed
    data = data.apply(pd.to_numeric, errors='coerce')

    # Updated interpolation/fill methods
    data.interpolate(method='linear', inplace=True)
    data.ffill(inplace=True)  # Replace fillna(method='ffill')
    data.bfill(inplace=True)  # Replace fillna(method='bfill')
    return data.values


def convert_time(path, usecols):
    df = pd.read_csv(path, usecols=usecols)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    offset = np.random.randint(-1100000, -1000000)
    df['timestamp'] = df['timestamp'].apply(convert_to_unixtime_with_offset, args=(offset,))
    df.interpolate(method='linear', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df.values
    

def build_dir(dataset_name):
    path = os.path.join(raw_path, dataset_name)
    
    base_dir = os.path.join(base_path, dataset_name)
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    
    export_dir = os.path.join(export_path_uts, dataset_name)
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)
        
    return path, base_dir, export_dir

def build_metric_dir(base_path, m_name):
    metric_dir = os.path.join(base_path, m_name)
    if not os.path.isdir(metric_dir):
        os.mkdir(metric_dir)
        
    return metric_dir

def check_valid(dataname):
    train_label_path = os.path.join(export_path_uts, dataname)
    for curve in os.listdir(train_label_path):
        labels = np.load(
            os.path.join(train_label_path, curve, "train_label.npy"),
            # allow_pickle=True
        )
        for i in labels:
            if i != 0 and i != 1:
                print("%s/%s train label has invalid value"%(dataname, curve))
                
    test_label_path = os.path.join(export_path_uts, dataname)
    for curve in os.listdir(test_label_path):
        labels = np.load(
            os.path.join(test_label_path, curve, "test_label.npy"),
            # allow_pickle=True
        )
        for i in labels:
            if i != 0 and i != 1:
                print("%s/%s test label has invalid value"%(dataname, curve))
    

# AIOPS
def AIOPS():
    path, base_dir, export_dir = build_dir("AIOPS")
    
    for test_name in os.listdir(path):
        if test_name[:4] == "test":
            train_name = 't' + test_name[4:]
        else:
            continue
        
        train_rawpath = os.path.join(path, train_name)
        test_rawpath = os.path.join(path, test_name)
        
        train_basepath = os.path.join(base_dir, train_name)
        test_basepath = os.path.join(base_dir, test_name)
        
        metric_dir = build_metric_dir(export_dir, train_name[1:-4])
        
        train_exportpath = os.path.join(metric_dir, "train.npy")
        test_exportpath = os.path.join(metric_dir, "test.npy")
        
        train_labelpath = os.path.join(metric_dir, "train_label.npy")
        test_labelpath = os.path.join(metric_dir, "test_label.npy")
        
        train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
        test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
        
        info_path = os.path.join(metric_dir, "info.json")
        
        print("--Processing dataset AIOPS %s--"%(test_name[4:]))
        
        min_interval_1 = insert_timestamp(train_rawpath, train_basepath, "timestamp")
        min_interval_2 = insert_timestamp(test_rawpath, test_basepath, "timestamp")
        assert min_interval_1 == min_interval_2
        
        # Training data
        train = fillna(train_basepath, usecols=["timestamp","value","label"])
        
        np.save(train_tspath, train[:,0])
        np.save(train_exportpath, train[:,1])
        # label would be invalid due to fillna
        train_label = np.floor(train[:,2])
        train_ano_ratio = np.count_nonzero(train_label >= 1) / len(train_label)
        np.save(train_labelpath, train_label)
        
        # Test data
        test = fillna(test_basepath, usecols=["timestamp","value","label"])
        
        np.save(test_tspath, test[:,0])
        np.save(test_exportpath, test[:,1])
        # label would be invalid due to fillna
        test_label = np.floor(test[:,2])
        test_ano_ratio = np.count_nonzero(test_label >= 1) / len(test_label)
        np.save(test_labelpath, test_label)
        
        total_ano_ratio = (np.count_nonzero(test_label >= 1) + np.count_nonzero(train_label >= 1)) / (len(test_label) + len(train_label))
        
        # Save info
        info = {
            'intervals' : int(min_interval_2),
            'training set anomaly ratio' : round(train_ano_ratio, 5),
            'testset anomaly ratio' : round(test_ano_ratio, 5),
            'total anomaly ratio' : round(total_ano_ratio, 5),
        }
        with open(info_path,'w') as f:
            json.dump(info, f, indent=4)
        

def NAB():
    path, base_dir, export_dir = build_dir("NAB")
    
    for dataname in os.listdir(path):
        data_rawpath = os.path.join(path, dataname)
        
        data_basepath = os.path.join(base_dir, dataname)
        
        metric_dir = build_metric_dir(export_dir, dataname)
        
        train_exportpath = os.path.join(metric_dir, "train.npy")
        test_exportpath = os.path.join(metric_dir, "test.npy")
        
        train_labelpath = os.path.join(metric_dir, "train_label.npy")
        test_labelpath = os.path.join(metric_dir, "test_label.npy")
        
        train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
        test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
        
        info_path = os.path.join(metric_dir, "info.json")
        
        print("--Processing dataset NAB %s--\n"%(dataname[:-3]))
        
        min_interval = insert_timestamp(data_rawpath, data_basepath, "timestamp")
        
        data = fillna(data_basepath, usecols=["timestamp","value","label"])
        
        # Split the train and test data by 5:5
        datalen = len(data)
        ratio = 0.5
        train = data[:int(datalen * ratio)]
        test = data[int(datalen * ratio):]
        
        # Training data
        np.save(train_tspath, train[:,0])
        np.save(train_exportpath, train[:,1])
        # label would be invalid due to fillna
        train_label = np.floor(train[:,2])
        train_ano_ratio = np.count_nonzero(train_label >= 1) / len(train_label)
        np.save(train_labelpath, train_label)
        
        # Test data
        np.save(test_tspath, test[:,0])
        np.save(test_exportpath, test[:,1])
        # label would be invalid due to fillna
        test_label = np.floor(test[:,2])
        test_ano_ratio = np.count_nonzero(test_label >= 1) / len(test_label)
        np.save(test_labelpath, test_label)
        
        total_ano_ratio = (np.count_nonzero(test_label >= 1) + np.count_nonzero(train_label >= 1)) / (len(test_label) + len(train_label))
        
        # Save info
        info = {
            'intervals' : int(min_interval),
            'training set anomaly ratio' : round(train_ano_ratio, 5),
            'testset anomaly ratio' : round(test_ano_ratio, 5),
            'total anomaly ratio' : round(total_ano_ratio, 5),
        }
        with open(info_path,'w') as f:
            json.dump(info, f, indent=4)        
        
     
def WSD():
    path, base_dir, export_dir = build_dir("WSD")
    
    for dataname in os.listdir(path):
        data_rawpath = os.path.join(path, dataname)
        
        data_basepath = os.path.join(base_dir, dataname)
        
        metric_dir = build_metric_dir(export_dir, dataname)
        
        train_exportpath = os.path.join(metric_dir, "train.npy")
        test_exportpath = os.path.join(metric_dir, "test.npy")
        
        train_labelpath = os.path.join(metric_dir, "train_label.npy")
        test_labelpath = os.path.join(metric_dir, "test_label.npy")
        
        train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
        test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
        
        info_path = os.path.join(metric_dir, "info.json")
        
        print("--Processing dataset WSD %s--"%(dataname[:-3]))
        
        min_interval = insert_timestamp(data_rawpath, data_basepath, "timestamp")
        
        data = fillna(data_basepath, usecols=["timestamp","value","label"])
        
        # Split the train and test data by 5:5
        datalen = len(data)
        ratio = 0.5
        train = data[:int(datalen * ratio)]
        test = data[int(datalen * ratio):]
        
        # Training data
        np.save(train_tspath, train[:,0])
        np.save(train_exportpath, train[:,1])
        # label would be invalid due to fillna
        train_label = np.floor(train[:,2])
        train_ano_ratio = np.count_nonzero(train_label >= 1) / len(train_label)
        np.save(train_labelpath, train_label)
        
        # Test data
        np.save(test_tspath, test[:,0])
        np.save(test_exportpath, test[:,1])
        # label would be invalid due to fillna
        test_label = np.floor(test[:,2])
        test_ano_ratio = np.count_nonzero(test_label >= 1) / len(test_label)
        np.save(test_labelpath, test_label)
        
        total_ano_ratio = (np.count_nonzero(test_label >= 1) + np.count_nonzero(train_label >= 1)) / (len(test_label) + len(train_label))
        
        # Save info
        info = {
            'intervals' : int(min_interval),
            'training set anomaly ratio' : round(train_ano_ratio, 5),
            'testset anomaly ratio' : round(test_ano_ratio, 5),
            'total anomaly ratio' : round(total_ano_ratio, 5),
        }
        with open(info_path,'w') as f:
            json.dump(info, f, indent=4)     
            
        print("--Processing End WSD %s--"%(dataname[:-3]))
            

def Yahoo():
    path, base_dir, export_dir = build_dir("Yahoo")
    
    for dataname in os.listdir(path):
        data_rawpath = os.path.join(path, dataname)
        
        data_basepath = os.path.join(base_dir, dataname)
        
        metric_dir = build_metric_dir(export_dir, dataname)
        
        train_exportpath = os.path.join(metric_dir, "train.npy")
        test_exportpath = os.path.join(metric_dir, "test.npy")
        
        train_labelpath = os.path.join(metric_dir, "train_label.npy")
        test_labelpath = os.path.join(metric_dir, "test_label.npy")
        
        train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
        test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
        
        info_path = os.path.join(metric_dir, "info.json")
        
        print("--Processing dataset Yahoo %s--"%(dataname[:-3]))
        
        min_interval = insert_timestamp(data_rawpath, data_basepath, "timestamp")
        
        data = fillna(data_basepath, usecols=["timestamp","value","label"])
        
        # Split the train and test data by 5:5
        datalen = len(data)
        ratio = 0.5
        train = data[:int(datalen * ratio)]
        test = data[int(datalen * ratio):]
        
        # Training data
        np.save(train_tspath, train[:,0])
        np.save(train_exportpath, train[:,1])
        # label would be invalid due to fillna
        train_label = np.floor(train[:,2])
        train_ano_ratio = np.count_nonzero(train_label >= 1) / len(train_label)
        np.save(train_labelpath, train_label)
        
        # Test data
        np.save(test_tspath, test[:,0])
        np.save(test_exportpath, test[:,1])
        # label would be invalid due to fillna
        test_label = np.floor(test[:,2])
        test_ano_ratio = np.count_nonzero(test_label >= 1) / len(test_label)
        np.save(test_labelpath, test_label)
        
        total_ano_ratio = (np.count_nonzero(test_label >= 1) + np.count_nonzero(train_label >= 1)) / (len(test_label) + len(train_label))
        
        # Save info
        info = {
            'intervals' : int(min_interval),
            'training set anomaly ratio' : round(train_ano_ratio, 5),
            'testset anomaly ratio' : round(test_ano_ratio, 5),
            'total anomaly ratio' : round(total_ano_ratio, 5),
        }
        with open(info_path,'w') as f:
            json.dump(info, f, indent=4)     
            
        print("--Processing End Yahoo %s--"%(dataname[:-3]))
        
def UCR():
    ori_path = "../UCR_Anomaly_FullData"
    export_dir = os.path.join(export_path_uts, "UCR")
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    for name in os.listdir(ori_path):
        
        curve = os.path.join(ori_path, name)
        name_split = name[:-4].split(sep="_")
        
        test_start = int(name_split[-3])
        ano_start = int(name_split[-2])
        ano_end = int(name_split[-1])
        curve_name = name_split[-4]
        
        metric_dir = os.path.join(export_dir, curve_name)
        if not os.path.isdir(metric_dir):
            os.mkdir(metric_dir)
        
        data = np.genfromtxt(curve)
        label = np.zeros(data.shape)
        label[ano_start:ano_end] = 1
        
        train = data[:test_start]
        test = data[test_start:]
        
        train_label = label[:test_start]
        test_label = label[test_start:]
        
        test_ano_ratio = (ano_end - ano_start) / len(test)
        total_ano_ratio = (ano_end - ano_start) / len(data)
        
        info = {
            'intervals' : 1,
            'training set anomaly ratio' : 0,
            'testset anomaly ratio' : round(test_ano_ratio, 5),
            'total anomaly ratio' : round(total_ano_ratio, 5),
        }
        
        train_exportpath = os.path.join(metric_dir, "train.npy")
        test_exportpath = os.path.join(metric_dir, "test.npy")
        
        train_labelpath = os.path.join(metric_dir, "train_label.npy")
        test_labelpath = os.path.join(metric_dir, "test_label.npy")
        
        train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
        test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
        
        info_path = os.path.join(metric_dir, "info.json")
        
        np.save(train_exportpath, train)
        np.save(test_exportpath, test)
        np.save(train_labelpath, train_label)
        np.save(test_labelpath, test_label)
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
            
    print("--Processing End UCR--")
        
def NEK():
    path, base_dir, export_dir = build_dir("NEK")
    
    for dataname in os.listdir(path):
        data_rawpath = os.path.join(path, dataname)
        
        data_basepath = os.path.join(base_dir, dataname)
        
        metric_dir = build_metric_dir(export_dir, dataname)
        
        train_exportpath = os.path.join(metric_dir, "train.npy")
        test_exportpath = os.path.join(metric_dir, "test.npy")
        
        train_labelpath = os.path.join(metric_dir, "train_label.npy")
        test_labelpath = os.path.join(metric_dir, "test_label.npy")
        
        train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
        test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
        
        info_path = os.path.join(metric_dir, "info.json")
        
        print("--Processing dataset NEK %s--"%(dataname[:-3]))
        
        # min_interval = insert_timestamp(data_rawpath, data_basepath, "timestamp")
        min_interval = 3600
        
        data = convert_time(data_basepath, usecols=["timestamp","value","label"])
        
        
        # Split the train and test data by 5:5
        datalen = len(data)
        ratio = 0.5
        train = data[:int(datalen * ratio)]
        test = data[int(datalen * ratio):]
        
        # Training data
        np.save(train_tspath, train[:,0])
        np.save(train_exportpath, train[:,1])
        # label would be invalid due to fillna
        train_label = np.floor(train[:,2])
        train_ano_ratio = np.count_nonzero(train_label >= 1) / len(train_label)
        np.save(train_labelpath, train_label)
        
        # Test data
        np.save(test_tspath, test[:,0])
        np.save(test_exportpath, test[:,1])
        # label would be invalid due to fillna
        test_label = np.floor(test[:,2])
        test_ano_ratio = np.count_nonzero(test_label >= 1) / len(test_label)
        np.save(test_labelpath, test_label)
        
        total_ano_ratio = (np.count_nonzero(test_label >= 1) + np.count_nonzero(train_label >= 1)) / (len(test_label) + len(train_label))
        
        # Save info
        info = {
            'intervals' : int(min_interval),
            'training set anomaly ratio' : round(train_ano_ratio, 5),
            'testset anomaly ratio' : round(test_ano_ratio, 5),
            'total anomaly ratio' : round(total_ano_ratio, 5),
        }
        with open(info_path,'w') as f:
            json.dump(info, f, indent=4)     
            
        print("--Processing End NEK %s--"%(dataname[:-3]))


def CalIt2():
    path, base_dir, export_dir = build_dir("CalIt2")

    for dataname in os.listdir(path):
        if not dataname.endswith('.csv'):
            continue  # Only process CSVs
        data_rawpath = os.path.join(path, dataname)
        data_basepath = os.path.join(base_dir, dataname)

        # Process both in_count and out_count as separate metrics
        for metric in ['in_count', 'out_count']:
            metric_dir = build_metric_dir(export_dir, f"{dataname[:-4]}_{metric}")

            train_exportpath = os.path.join(metric_dir, "train.npy")
            test_exportpath = os.path.join(metric_dir, "test.npy")
            train_labelpath = os.path.join(metric_dir, "train_label.npy")
            test_labelpath = os.path.join(metric_dir, "test_label.npy")
            train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
            test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
            info_path = os.path.join(metric_dir, "info.json")

            print(f"--Processing CalIt2 {metric}--")

            # Process timestamps (30 minute intervals in sample data)
            min_interval = insert_timestamp(data_rawpath, data_basepath, "timestamp")

            # Load specific metric columns: timestamp, [metric], is_anomaly
            data = fillna(data_basepath, usecols=["timestamp", metric, "is_anomaly"])

            # 50:50 split like other datasets
            datalen = len(data)
            ratio = 0.5
            train = data[:int(datalen * ratio)]
            test = data[int(datalen * ratio):]

            # Training data
            np.save(train_tspath, train[:, 0])  # timestamp
            np.save(train_exportpath, train[:, 1])  # metric value
            train_label = np.floor(train[:, 2])  # ensure binary labels
            train_ano_ratio = np.count_nonzero(train_label >= 1) / len(train_label)
            np.save(train_labelpath, train_label)

            # Test data
            np.save(test_tspath, test[:, 0])
            np.save(test_exportpath, test[:, 1])
            test_label = np.floor(test[:, 2])
            test_ano_ratio = np.count_nonzero(test_label >= 1) / len(test_label)
            np.save(test_labelpath, test_label)

            # Metadata
            total_ano_ratio = (np.count_nonzero(test_label) + np.count_nonzero(train_label)) / (
                        len(test_label) + len(train_label))
            info = {
                'intervals': int(min_interval),
                'training_set_anomaly_ratio': round(train_ano_ratio, 5),
                'testset_anomaly_ratio': round(test_ano_ratio, 5),
                'total_anomaly_ratio': round(total_ano_ratio, 5),
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=4)


def NormA():
    path, base_dir, export_dir = build_dir("NormA")

    for dataname in os.listdir(path):
        if not dataname.endswith('.csv'):
            continue  # Skip non-CSV files

        data_rawpath = os.path.join(path, dataname)
        data_basepath = os.path.join(base_dir, dataname)

        metric_dir = build_metric_dir(export_dir, dataname[:-4])

        train_exportpath = os.path.join(metric_dir, "train.npy")
        test_exportpath = os.path.join(metric_dir, "test.npy")
        train_labelpath = os.path.join(metric_dir, "train_label.npy")
        test_labelpath = os.path.join(metric_dir, "test_label.npy")
        train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
        test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
        info_path = os.path.join(metric_dir, "info.json")

        print(f"--Processing NormA dataset: {dataname[:-4]}--")

        # Process timestamps (already normalized integers)
        min_interval = insert_timestamp(data_rawpath, data_basepath, "timestamp")

        # Load data with specified columns
        data = fillna(data_basepath, usecols=["timestamp", "value", "is_anomaly"])

        # 50:50 split like other datasets
        datalen = len(data)
        ratio = 0.5
        train = data[:int(datalen * ratio)]
        test = data[int(datalen * ratio):]

        # Training data
        np.save(train_tspath, train[:, 0])  # timestamp
        np.save(train_exportpath, train[:, 1])  # value
        train_label = np.floor(train[:, 2]).astype(np.int8)  # force integer labels
        train_ano_ratio = np.count_nonzero(train_label) / len(train_label)
        np.save(train_labelpath, train_label)

        # Test data
        np.save(test_tspath, test[:, 0])
        np.save(test_exportpath, test[:, 1])
        test_label = np.floor(test[:, 2]).astype(np.int8)
        test_ano_ratio = np.count_nonzero(test_label) / len(test_label)
        np.save(test_labelpath, test_label)

        # Metadata
        total_ano_ratio = (np.count_nonzero(test_label) + np.count_nonzero(train_label)) / (
                    len(test_label) + len(train_label))
        info = {
            'intervals': int(min_interval),
            'training_set_anomaly_ratio': round(train_ano_ratio, 5),
            'testset_anomaly_ratio': round(test_ano_ratio, 5),
            'total_anomaly_ratio': round(total_ano_ratio, 5),
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)


def GutenTAG():
    path, base_dir, export_dir = build_dir("GutenTAG")

    for dataname in os.listdir(path):
        dir_path = os.path.join(path, dataname)
        if not os.path.isdir(dir_path):
            continue
        if dataname == "multivariates":
            continue  # Skip the multivariates directory

        # Process train_anomaly.csv as before (if needed)
        train_csv = os.path.join(dir_path, "train_anomaly.csv")
        if os.path.isfile(train_csv):
            train_basepath = os.path.join(base_dir, f"{dataname}_train_anomaly.csv")
            metric_dir = build_metric_dir(export_dir, dataname)
            train_exportpath = os.path.join(metric_dir, "train.npy")
            train_labelpath = os.path.join(metric_dir, "train_label.npy")
            train_tspath = os.path.join(metric_dir, "train_timestamp.npy")
            info_path = os.path.join(metric_dir, "info.json")
            print(f"--Processing GutenTAG train {dataname}--")
            min_interval = insert_timestamp(train_csv, train_basepath, "timestamp")
            cols = list(pd.read_csv(train_basepath, nrows=1).columns)
            value_cols = [col for col in cols if col not in ["timestamp", "is_anomaly"]]
            usecols = ["timestamp"] + value_cols + ["is_anomaly"]
            data = fillna(train_basepath, usecols=usecols)
            np.save(train_tspath, data[:, 0])
            np.save(train_exportpath, data[:, 1:-1])
            labels = np.floor(data[:, -1])
            np.save(train_labelpath, labels)
            train_ano_ratio = np.count_nonzero(labels >= 1) / len(labels)
            # Save info (will update later if test.csv exists)
            info = {
                'intervals': int(min_interval),
                'training_set_anomaly_ratio': round(train_ano_ratio, 5),
            }
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=4)

        # Now process test.csv
        test_csv = os.path.join(dir_path, "test.csv")
        if os.path.isfile(test_csv):
            test_basepath = os.path.join(base_dir, f"{dataname}_test.csv")
            metric_dir = build_metric_dir(export_dir, dataname)
            test_exportpath = os.path.join(metric_dir, "test.npy")
            test_labelpath = os.path.join(metric_dir, "test_label.npy")
            test_tspath = os.path.join(metric_dir, "test_timestamp.npy")
            info_path = os.path.join(metric_dir, "info.json")
            print(f"--Processing GutenTAG test {dataname}--")
            min_interval = insert_timestamp(test_csv, test_basepath, "timestamp")
            cols = list(pd.read_csv(test_basepath, nrows=1).columns)
            value_cols = [col for col in cols if col not in ["timestamp", "is_anomaly"]]
            usecols = ["timestamp"] + value_cols + ["is_anomaly"]
            data = fillna(test_basepath, usecols=usecols)
            np.save(test_tspath, data[:, 0])
            np.save(test_exportpath, data[:, 1:-1])
            labels = np.floor(data[:, -1])
            np.save(test_labelpath, labels)
            test_ano_ratio = np.count_nonzero(labels >= 1) / len(labels)
            # Update info.json with test anomaly ratio if exists
            if os.path.isfile(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
            else:
                info = {'intervals': int(min_interval)}
            info['testset_anomaly_ratio'] = round(test_ano_ratio, 5)
            if 'training_set_anomaly_ratio' in info:
                total_ano_ratio = info['training_set_anomaly_ratio']
                total_ano_ratio = (total_ano_ratio + test_ano_ratio) / 2
                info['total_anomaly_ratio'] = round(total_ano_ratio, 5)
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"--Finished processing GutenTAG test {dataname}--")


if __name__ == '__main__':
    # AIOPS()
    # NAB()
    # WSD()
    # Yahoo()
    # check_valid("AIOPS")
    # check_valid("NAB")
    # check_valid("WSD")
    # check_valid("Yahoo")
    
    # UCR()
    # check_valid("UCR")
    # NEK()
    # check_valid("NEK")

    # Novel datasets preprocessing
    CalIt2()
    check_valid("CalIt2")
    #
    # NormA()
    # check_valid("NormA")
    #
    # GutenTAG()
    # check_valid("GutenTAG")