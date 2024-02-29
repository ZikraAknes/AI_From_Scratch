'''Made by: Zikra Fathirizqi Aknes'''

import sys
import time
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    MAGENTA = '\033[35m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Loading():
    def __init__(self):
        self.bar_length = 0
        self.elapsed_time = []
        self.start_time = 0
        self.iteration = 0
        self.var_length = 0
        self.loss = 0
        self.acc = 0

    def start_loading_bar(self, variable):
        self.start_time = time.time()
        self.var_length = variable
        self.loss = 0
        self.acc = 0
        cols, rows = os.get_terminal_size()
        self.iteration = 0
        self.elapsed_time.clear()
        time_str = " [00:00<?, ?/it]"
        iterations_str = " " + str(self.iteration) + "/" + str(self.var_length)
        self.bar_length = cols - len(time_str) - len(iterations_str) - 6
        percentage, bar, bar_end = self.create_bar()
        
        print("\n\n")
        self.print_bar(time_str, percentage, bar, bar_end, iterations_str)

        return variable

    def iterate_loading_bar(self, loss, acc):
        self.loss = "{:.4f}".format(loss)
        self.acc = "{:.4f}".format(acc)

        cols, rows = os.get_terminal_size()
        
        hour = "?"
        minute = "?"
        second = "?"

        self.elapsed_time.append(time.time() - self.start_time)
        self.start_time = time.time()

        pred_time = (self.elapsed_time[self.iteration-1]*(self.var_length - self.iteration))

        hour_pred, min_pred, sec_pred = self.time_converter(pred_time)

        hour, minute, second = self.time_converter(sum(self.elapsed_time))

        elapsed_str = str(hour).zfill(2) + ":" + str(minute).zfill(2) + ":" + str(second).zfill(2) + "<"
        pred_str = str(hour_pred).zfill(2) + ":" + str(min_pred).zfill(2) + ":" + str(sec_pred).zfill(2) + ", "

        if self.elapsed_time[self.iteration-1]*self.var_length < 3600:
            elapsed_str = elapsed_str[3:]
            pred_str = pred_str[3:]

        itr_time_str = str("{:.2f}".format(self.elapsed_time[self.iteration-1])) + "s/it"
        time_str = " [" + elapsed_str + pred_str + itr_time_str + "]"
        iterations_str = " " + str(self.iteration) + "/" + str(self.var_length)

        self.bar_length = cols - len(time_str) - len(iterations_str) - 6

        percentage, bar, bar_end = self.create_bar()

        self.print_bar(time_str, percentage, bar, bar_end, iterations_str)
        
    def time_converter(self, elapsed_time):
        time_temp = int(elapsed_time)

        hour = time_temp//3600
        time_temp -= hour*3600 
        minute = time_temp//60
        time_temp -= minute*60
        sec = time_temp 

        return int(hour), int(minute), int(sec)
    
    def create_bar(self):
        bar_end = ""
        bar_start = ""

        percentage = '{:3d}'.format(int((100/self.var_length)*self.iteration))
        
        total_bar = int((self.bar_length/self.var_length)*self.iteration)

        for i in range(total_bar):
            bar_start += "━"

        for i in range(self.bar_length - (len(bar_start))):
            bar_end += "━"

        return percentage, bar_start, bar_end

    def print_bar(self, time_str, percentage, bar_start, bar_end, iterations_str):
        UP = "\x1B[3F"
        DOWN = "\x1B[1E"
        CLR = "\x1B[2K"

        if self.iteration == self.var_length:
            bar_color = bcolors.OKGREEN
        else:
            bar_color = bcolors.FAIL

        load_bar = f" {bar_color}" + bar_start + f"{bcolors.ENDC}" +  bar_end

        sys.stdout.write(f"{UP}{CLR}{bar_color}{percentage}%{bcolors.ENDC}{load_bar}{iterations_str}{bcolors.ENDC}{time_str}{DOWN}{CLR}\tAverage Loss:\tAverage Accuracy:{DOWN}{CLR}\t{self.loss}\t\t{self.acc}{DOWN}")
        
        sys.stdout.flush()
        self.iteration += 1