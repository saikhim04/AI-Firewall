import datetime

with open("decision_log.txt", "a") as f:
    log_entry = f"{datetime.datetime.now()} | {src_ip} -> {dst_ip} | {decision}\n"
    f.write(log_entry)