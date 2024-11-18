import matplotlib.pyplot as plt

# Data from tests
cpu_sizes = [100, 1000, 10000, 100000, 1000000]
cpu_times = [0.0016378, 0.0201198, 1.75837, 178.093, 18341.3]
cuda_times = [0.345546, 0.00520373, 0.159758, 16.3282, 1332.92801]
speedup_factors = [0.00473975, 3.86642, 11.0065, 10.9071, 13.76]

# Plot CPU and GPU times
plt.figure(figsize=(12, 6))
plt.plot(cpu_sizes, cpu_times, label="CPU Time (s)", marker='o')
plt.plot(cpu_sizes, cuda_times, label="GPU Time (s)", marker='o')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Rows (Log Scale)")
plt.ylabel("Time (Seconds, Log Scale)")
plt.title("CPU vs GPU Times for Inner Join")
plt.legend()
plt.grid(True)
plt.show()

# Plot speedup factor
plt.figure(figsize=(12, 6))
plt.plot(cpu_sizes, speedup_factors, label="GPU Speedup Factor", marker='o', color='green')
plt.xscale("log")
plt.xlabel("Number of Rows (Log Scale)")
plt.ylabel("Speedup Factor (GPU/CPU)")
plt.title("GPU Speedup Factor vs Dataset Size")
plt.legend()
plt.grid(True)
plt.show()