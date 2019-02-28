import datetime

log_filepath = "process.log"


def write_to_logfile(text):
    with open(log_filepath, "a") as f_:
        f_.write(text)


def log(process_function):
    def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.utcnow()

        entry_str = timestamp.strftime("utc %Y-%m-%d %H:%M:%S ") \
            .join(process_function.im_class.__name__) \
            .join(".") \
            .join(process_function.__name__) \
            .join(str(args[1:])) \
            .join(str(kwargs)) \
            .join("\n")

        write_to_logfile(entry_str)
        return process_function(*args, **kwargs)
    return wrapper
