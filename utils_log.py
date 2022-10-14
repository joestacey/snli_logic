import os
from datetime import datetime

# Useful functions to be used:


class Log():

    def __init__(self, log_name_, params_training, print_results=True):

        self.log_name = log_name_
        self.params = params_training
        self.print_results = print_results
        self.opening_message()


    def msg(self, input_list):
        """
        Parameters:
        ---------

        input_list : list
            A list containing different strings that should be printed to log file
        """

        log_file = open(self.log_name, 'a')

        for msg in input_list:
            log_file.write(msg + "\n")
            if self.print_results:
                print(msg + "\n")

        log_file.write("\n \n")
        if self.print_results:
            print("\n \n")

        log_file.close()

        
    def opening_message(self):
        """ 
        Enter the PID and time/date when log file created
        """
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        log_file = open(self.log_name, 'a')
        
        log_file.write("\n\n-----\n")
        log_file.write(dt_string + "\n")
        if self.print_results:
            print(dt_string + "\n")
        log_file.write(str(os.getpid()) + " is pid \n")
        
        for parameter in vars(self.params):
            if self.print_results:
                print(parameter, " - ",  getattr(self.params, parameter))
            log_file.write(parameter + " - " +  str(getattr(self.params, parameter)) + "\n")

        if self.print_results:
            print(str(os.getpid()) + " is pid \n \n")
        log_file.close()


