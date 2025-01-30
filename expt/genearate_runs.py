# Define the parameters for the combinations
THEORYCLASS = ["VFAE", "VFAESI", "VFBAESI"]
GAMMA = [0.1, 0.5, 0.9, 1.0]
RUN = range(0, 10)  # 1 to 10 inclusive

# Open the file in write mode
with open("runs.txt", "w") as file:
    # Iterate over all combinations of THEORYCLASS, GAMMA, and RUN
    for theory_class in THEORYCLASS:
        for gamma in GAMMA:
            for run in RUN:
                # Write the formatted command line to the file
                file.write(f"python vae-expt.py --device cuda:0 --epochs 500 --theoryclass {theory_class} --gamma {gamma} --run {run}\n")

print("runs.txt has been generated.")


