import re
import pandas as pd
import random

rules = []
with open("firewall_rules.txt") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("-A"):
            continue
        
        proto = re.search(r"-p (\w+)", line)
        proto = proto.group(1) if proto else "any"
        
        dport = re.search(r"--dport (\d+)", line)
        dport = int(dport.group(1)) if dport else 0
        
        action = re.search(r"-j (\w+)", line)
        action = 1 if action and action.group(1)=="ACCEPT" else 0

        rules.append({"proto":proto, "dport":dport, "action":action})

# สร้าง dataset จาก rule
dataset = []
proto_map = {"tcp":6, "udp":17, "icmp":1, "any":0}
for rule in rules:
    for _ in range(200):  # generate sample หลาย packet ต่อ rule
        proto = proto_map[rule["proto"]]
        sport = random.randint(1024,65535)
        dport = rule["dport"]
        srcIP = random.randint(0,2**32-1)   # สุ่ม IP
        dstIP = random.randint(0,2**32-1)
        dataset.append([proto, sport, dport, srcIP, dstIP, rule["action"]])

df = pd.DataFrame(dataset, columns=["proto","sport","dport","srcIP","dstIP","label"])
df.to_csv("ai_firewall_dataset.csv", index=False)
print("Dataset generated:", df.shape)
