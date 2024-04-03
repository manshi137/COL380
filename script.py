import os

# List of values for n and t
n_values = [2100, 2800, 3500, 4200, 4900]
# n_values = [5600]
t_values = [2, 4, 6, 8, 10]

# Path to the executable
executable_path = "./a.out"

# Compile the code if not already compiled
os.system("g++ pthreadnew.cpp -std=c++11")

# Iterate over all combinations of n and t
for n in n_values:
    for t in t_values:
        # Run the executable with arguments n and t
        os.system(f"{executable_path} {n} {t}")
