import csv
import re

# ไฟล์ input ที่ export มาจาก iptables
input_file = "firewall_rules.txt"
output_file = "firewall_rules.csv"

# regex pattern สำหรับ parse rule
rule_pattern = re.compile(
    r"-A\s+\S+"                # chain name
    r"(?:\s+-p\s+(\S+))?"      # protocol
    r"(?:\s+-s\s+(\S+))?"      # source IP
    r"(?:\s+-d\s+(\S+))?"      # destination IP
    r"(?:\s+--sport\s+(\d+))?" # source port
    r"(?:\s+--dport\s+(\d+))?" # destination port
    r".*?-j\s+(\S+)"           # action (ACCEPT/DROP/REJECT)
)

# เขียนเป็น CSV
with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["protocol", "src_ip", "dst_ip", "src_port", "dst_port", "action"])

    for line in f_in:
        match = rule_pattern.search(line)
        if match:
            protocol, src_ip, dst_ip, sport, dport, action = match.groups()
            writer.writerow([
                protocol if protocol else "any",
                src_ip if src_ip else "any",
                dst_ip if dst_ip else "any",
                sport if sport else "any",
                dport if dport else "any",
                action
            ])

print(f"Done! Exported training set to {output_file}")