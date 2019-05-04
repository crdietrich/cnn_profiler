"""Parse data from the Pstats output

Colin Dietrich 2019
"""

import io
import pstats
import pandas as pd


class PstatsParse:
    def __init__(self, filepath):
        
        # basic pstats info
        self.filepath = filepath
        self.p = pstats.Stats(self.filepath)
        self.total_time_s = self.p.total_tt
        self.total_time_min = self.total_time_s / 60.0
        
        fs = self.filepath.split('_')
        self.platform = fs[0]
        self.model_type = fs[1]
        self.run_id = fs[2]
        
        fspred = self.filepath.split('_pstats.txt')
        self.df_pred = pd.read_csv(fspred[0] + '_predictions.csv')
        
        n = len(self.df_pred)
        self.acc = self.df_pred.y_pred == self.df_pred.y_true
        self.acc = self.acc.sum()/n
        self.acc_dog = self.df_pred.y_pred_dog.sum()/n
        
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
