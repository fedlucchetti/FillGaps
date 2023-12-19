import shutil

class MessagePrinter:
    COLORS = {
        "SUCCESS": "\033[92m",  # Green
        "ERROR": "\033[91m",    # Red
        "BUG": "\033[95m",      # Magenta
        "WARNING": "\033[93m",  # Yellow
        "FAILURE": "\033[91m",  # Red
        "INFO": "\033[94m",     # Blue
        "ENDC": "\033[0m"       # Reset to default color
    }
    def __init__(self, verbose=True):
        self.verbose = verbose

    def concatenate(self,*messages):
        return ' '.join(str(message) for message in messages)
    
    def success(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            print(f"\033[92m [ SUCCESS ] {message} \033[0m ")

    def error(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            print(f"\033[95m [  ERROR  ] {message} \033[0m ")

    def warning(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            print(f"\033[38;5;208m [ WARNING ] {message} \033[0m ")

    def failure(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            print(f"\033[91m [ FAILURE ] {message} \033[0m ")

    def info(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            print(f"\033[94m [   INFO  ] {message} \033[0m ")

    def separator(self):
        if self.verbose:
            columns, _ = shutil.get_terminal_size(fallback=(80, 20))
            print("\n" + f" #" * columns," \033[0m")

if __name__=="__main__":
    printer = MessagePrinter()
    value = 23
    message="is the answer to the ultimate question of the Universe."
    printer.separator()
    printer.success(value,message)
    printer.error(value,message)
    printer.warning(value,message)
    printer.failure(value,message)
    printer.info(value,message)
    printer.separator()