"""Parse data from the Pstats output

Colin Dietrich 2019
"""

import os
import io
import json
import pstats
import pandas as pd

import config


def collate(data_directory=config.data_directory):
    """Collate all files needed for pstats and power profiling

    Parameters
    ----------
    data_directory : str, path to directory with power and profile data files

    Returns
    -------
    power_files : list of lists, containing len=2 items:
        power_file : str, path to power file for the data id
        profile_file : str, path to profile timing file for data id
    """
    
    files = []
    for (dirpath, dirnames, filenames) in os.walk(config.data_directory):
        files.extend(filenames)
        break
    set_samples = set([x.split(" - ")[0] for x in files])

    power_files = []
    profile_files = []

    for ss in set_samples:
        power_sample = [None,None]
        for f in files:
            if ss in f:
                if 'power' in f:
                    power_sample[0] = f
                if 'profile_output' in f:
                    power_sample[1] = f
                if 'pstats' in f:
                    profile_files.append(f)
                if 'predictions' in f:
                    pass
        power_files.append(power_sample)
    return power_files, profile_files

def csv_resource(fp):
    """Parse a .csv file generated with Meerkat
    
    Parameters
    ----------
    fp : filepath to saved data
    
    Returns
    -------
    meta : dict, metadata describing data
    df : Pandas DataFrame, data recorded from device(s) described in meta
    """

    with open(fp, 'r') as f:
        sbang = f.readline()
    _meta = json.loads(sbang[2:])
    _df = pd.read_csv(fp,
                     delimiter=_meta['delimiter'],
                     comment=_meta['comment'])
    _df['datetime64_ns'] = pd.to_datetime(_df[_meta['time_format']])

    return _meta, _df

def calc_W_h(_df, t0, t1):
    """Calculate Watt hours of energy consumed

    Parameters
    ----------
    _df : Pandas DataFrame
    t0 : datetime object, time of start of test
    t1 : datetime object, time of end of test
    
    Returns
    -------
    Wh : float, Watt-hours of energy consumed during test
    dt : Timedelta, time spent conducting test
    W_mean : float, mean Watt usage
    W_max : float, max Watt use
    W_min : float, min Wat use
    W_mean_off : float, mean Wattage at idle
    """
    
    _df['dt'] = _df.datetime64_ns.diff()
    _df['dts'] = _df.dt.dt.seconds + (_df.dt.dt.microseconds / 1000000)
    _df['W_s'] = _df.watts * _df.dts
    
    _df_in = _df.loc[(_df.datetime64_ns > t0) & 
                 (_df.datetime64_ns < t1)]
    W_s = _df_in.W_s.sum()
    dt = _df_in.datetime64_ns.max() - _df_in.datetime64_ns.min()
    W_mean = _df_in.watts.mean()
    W_max = _df_in.watts.max()
    W_min = _df_in.watts.min()
    W_h = W_s / (60 * 60)
    
    _df_out = _df[(_df.datetime64_ns < t0) | 
                 (_df.datetime64_ns > t1)]
    W_mean_off = _df_out.watts.mean()    
    return W_h, dt, W_mean, W_max, W_min, W_mean_off

def process(csv_file, timing_file):
    _meta, _df = csv_resource(config.data_directory + csv_file)
    _df['watts'] = _df.voltage * _df.current
    
    with open(config.data_directory + os.path.sep + timing_file, 'r') as f:
        _start = f.readline().split(',')[0]
        _end = f.readline().split(',')[0]
        _start = pd.to_datetime(_start)
        _end = pd.to_datetime(_end)
        
    _W_h, dt, W_mean, W_max, W_min, W_mean_off = calc_W_h(_df, _start, _end)
    return _meta, _df, _start, _end, _W_h, dt, W_mean, W_max, W_min, W_mean_off

def plot_profile(df, t0, t1):
    df[['datetime64_ns', 'watts']].plot(x='datetime64_ns');
    plt.vlines(t0, 0, 5, colors='green')
    plt.vlines(t1, 0, 5, colors='red');

def pstats_compile(profile_files):
    data = []
    for f in profile_files:
        fp = os.path.normpath(config.data_directory) + os.path.sep + f
        ps = Pstats(fp)
        a = [ps.total_time_min, ps.platform, ps.pu, ps.pu_type, ps.tf_version, ps.run_id,
             ps.acc, ps.acc_dog, ps.filepath]
        data.append(a)
    cols = ["time_min", "platform", "PU", "PU_type", "TF_version", "run_id", 
            "acc", "acc_dog", "filepath"]
    return pd.DataFrame(data, columns=cols)

def power_compile(power_files):
    """Compile power profile data"""
    power_data = []
    for power_csv, profile_txt in power_files:
        if power_csv is not None:
            fid = power_csv.split(" - ")
            fid = fid[0].split("_")
            (meta, df, t0, t1, w_h, dt, 
             W_mean, W_max, W_min, W_mean_off) = process(power_csv, 
                                                         profile_txt)
            power_data.append(fid + [dt, w_h, W_mean, W_max, W_min, W_mean_off])

    columns = ["platform", "PU", "PU_type", "TF_version", "run_id", "power_time", 
               "Watt_hours", "Watts_mean", "Watts_max", "Watts_min", "Watts_mean_off"]
    _df = pd.DataFrame(power_data, columns=columns)
    return _df

class Pstats(object):
    def __init__(self, filepath):
        
        # basic pstats info
        self.filepath = filepath
        self.p = pstats.Stats(self.filepath)
        self.total_time_s = self.p.total_tt
        self.total_time_min = self.total_time_s / 60.0
        
        fspred = self.filepath.split(' - pstats.txt')
        self.df_pred = pd.read_csv(fspred[0] + ' - predictions.csv')
        
        fp = fspred[0].split(os.path.sep)
        fs = fp[-1].split('_')
        self.platform = fs[0]
        self.pu = fs[1]
        self.pu_type = fs[2]
        self.tf_version = fs[3]
        self.run_id = fs[4]
        
        # accuracy
        n = len(self.df_pred)
        self.acc = self.df_pred.y_pred == self.df_pred.y_true
        self.acc = self.acc.sum()/n
        self.acc_dog = self.df_pred.y_pred_dog.sum()/n
        
        # precision
        
        # capture print output
        self.sio = io.StringIO()
        self.stats = pstats.Stats(self.filepath, stream=self.sio)
        self.stats.print_stats();
        self.data = self.sio.getvalue().split("\n")
        
        # store parsed data in list and dataframe
        self.column_names = ['ncalls',
                             'tottime',
                             'percall',
                             'cumtime',
                             'percall',
                             'filename:lineno(function)']

        self.a = []
        self.df = pd.DataFrame()
        self.parse_list(self.data)
        
    def parse_list(self, data_list):
        """Parse the list of data from pstats print method"""
        r = False
        for line in data_list:
            if len(line) == 0:
                continue
            if 'ncalls' in line:
                r = True
            if not r:
                continue
            line_list = line.split()
            if "/" in line_list[0]:
                x0 = line_list[0].split("/")
                line_list = [x0[0]] + line_list[1:]
            self.a.append(line_list[0:6])
        self.df = pd.DataFrame(self.a[1:], 
                               columns=self.column_names,
                               dtype=float)
        self.df.sort_values('cumtime', ascending=False, inplace=True)
        
    def calc_time(self, filename_lineno):
        """Find the time spent on a specific filename and
        line number
        
        Example filename_lineno might be:
        "tensorflow_backend.py:2696(__call__)"
        
        """
        cn = "filename:lineno(function)"
        return self.df[self.df[cn] == filename_lineno].cumtime.values[0]


