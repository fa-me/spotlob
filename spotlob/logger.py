import datetime

log_filepath = "process.log"


def write_to_logfile(text):
    with open(log_filepath, "a") as f_:
        f_.write(text)


def log(process_function):
    def wrapper(*args, **kwargs):
        timestr = datetime.datetime.utcnow().strftime("utc %Y-%m-%d %H:%M:%S ")
        write_to_logfile(timestr + process_function.im_class.__name__ +
                         "." + process_function.__name__ + str(args[1:]) + str(kwargs)+"\n")
        return process_function(*args, **kwargs)
    return wrapper
