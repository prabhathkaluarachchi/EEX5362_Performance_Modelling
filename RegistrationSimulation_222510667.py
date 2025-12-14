# RegistrationSimulation_222510667.py
# University Registration Queue Simulation using SimPy
# EEX5362 – Performance Modelling

import os
import random
import simpy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ----------------------------
# PRE SETTINGS
# ----------------------------
NUM_STUDENTS = 198
SERVICE_MIN_SEC = 300   # 5 minutes
SERVICE_MAX_SEC = 600   # 10 minutes
OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)

# ----------------------------
# REAL DATASET LOADING
# ----------------------------
def load_or_generate_data(csv_name="queue_data.csv", n=NUM_STUDENTS):

    print("\nLoading dataset from", csv_name, "\n")

    df = pd.read_csv(csv_name)

    # parse arrival time (DD-MM-YYYY H.MM)
    df["arrival_time"] = pd.to_datetime(
        df["arrival_time"],
        format="%d-%m-%Y %H.%M",
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["arrival_time"])

    # create student IDs
    df["student_id"] = [f"s220{10000+i:05d}" for i in range(len(df))]

    # generate realistic service times
    df["service_time_sec"] = np.random.randint(
        SERVICE_MIN_SEC, SERVICE_MAX_SEC + 1, size=len(df)
    )

    return df.head(n).copy()

# ----------------------------
# STUDENT PROCESSEMENT
# ----------------------------
def student_process(env, student, counters, records):

    yield env.timeout(student["arrival_time_sim"] - env.now)

    t_req = env.now
    with counters.request() as req:
        yield req

        wait = env.now - t_req

        records.append({
            "student_id": student["student_id"],
            "arrival_time": student["arrival_time"],
            "service_start": env.now,
            "wait_sec": int(wait),
            "service_time_sec": int(student["service_time_sec"])
        })

        yield env.timeout(student["service_time_sec"])

# ----------------------------
# RUN SCENARIO
# ----------------------------
def run_scenario(df_students, num_counters):

    df = df_students.copy().reset_index(drop=True)

    base_time = df["arrival_time"].min()
    df["arrival_time_sim"] = (
        df["arrival_time"] - base_time
    ).dt.total_seconds()

    env = simpy.Environment()
    counters = simpy.Resource(env, capacity=num_counters)
    records = []

    for _, row in df.iterrows():
        env.process(student_process(env, row, counters, records))

    env.run()

    stats_df = pd.DataFrame(records)

    total_time_sec = stats_df["service_start"].max()
    num_served = len(stats_df)

    throughput = num_served / (total_time_sec / 3600)
    avg_wait = stats_df["wait_sec"].mean()
    max_wait = stats_df["wait_sec"].max()

    total_service = stats_df["service_time_sec"].sum()
    utilization = (total_service / (num_counters * total_time_sec)) * 100

    return {
        "num_counters": num_counters,
        "throughput_per_hr": throughput,
        "avg_wait_sec": avg_wait,
        "max_wait_sec": max_wait,
        "utilization_pct": utilization,
        "num_served": num_served,
        "details": stats_df
    }

# ----------------------------
# RUN EXPERIMENTS
# ----------------------------
def run_experiments():

    df = load_or_generate_data()

    scenarios = [1, 2, 4] # Number of counters to simulate
    results = []

    for n in scenarios:
        print(f"Running scenario with {n} counters...")
        res = run_scenario(df, n)
        results.append(res)

        res["details"].to_csv(
            os.path.join(OUTPUT_DIR, f"details_{n}_counters.csv"),
            index=False
        )

    summary = pd.DataFrame([{
        "counters": r["num_counters"],
        "throughput_per_hr": r["throughput_per_hr"],
        "avg_wait_min": r["avg_wait_sec"] / 60,
        "max_wait_min": r["max_wait_sec"] / 60,
        "utilization_pct": r["utilization_pct"],
        "num_served": r["num_served"]
    } for r in results])

    print("\nSimulation Completed. Summary of Results:\n")
    print(summary.to_string(index=False))

    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

    # ----------------------------
    # PLOTS USED FOR ANALYSIS
    # ----------------------------

    # Throughput
    plt.figure(figsize=(8,5))
    plt.bar(summary["counters"].astype(str),
            summary["throughput_per_hr"],
            color="skyblue", edgecolor="black")
    plt.xlabel("Number of Counters")
    plt.ylabel("Throughput (students/hr)")
    plt.title("Throughput by Number of Counters")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "throughput_by_counters.png"))
    plt.close()

    # Average Wait Time
    plt.figure(figsize=(8,5))
    plt.plot(summary["counters"],
             summary["avg_wait_min"],
             marker="o", linewidth=2, color="orange")
    plt.xlabel("Number of Counters")
    plt.ylabel("Average Wait (minutes)")
    plt.title("Average Waiting Time")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(scenarios)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_wait_by_counters.png"))
    plt.close()

    # Max Wait Time
    plt.figure(figsize=(8,5))
    plt.plot(summary["counters"],
             summary["max_wait_min"],
             marker="o", linewidth=2, color="red")
    plt.xlabel("Number of Counters")
    plt.ylabel("Max Wait (minutes)")
    plt.title("Maximum Waiting Time")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(scenarios)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "max_wait_by_counters.png"))
    plt.close()

    # Utilization
    plt.figure(figsize=(8,5))
    plt.bar(summary["counters"].astype(str),
            summary["utilization_pct"],
            color="green", edgecolor="black", alpha=0.7)
    plt.xlabel("Number of Counters")
    plt.ylabel("Utilization (%)")
    plt.title("Counter Utilization")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "utilization_by_counters.png"))
    plt.close()

    print("\nAll outputs saved in folder:")
    print(os.path.abspath(OUTPUT_DIR) + "\n")

# ---------------------------- 
# MAIN 
# ----------------------------
if __name__ == "__main__":
    run_experiments()
