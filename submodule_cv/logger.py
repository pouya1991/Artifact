import sys

class Logger(object):
    def __init__(self, log_loc):
        self.terminal = sys.stdout
        self.log = open(log_loc, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()
        self.terminal.flush()