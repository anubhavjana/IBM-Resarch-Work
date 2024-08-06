# import matplotlib.pyplot as plt

# # Data for Mean Queue Size (Updated Prompt)
# requests_per_minute = [60, 70, 80, 90, 100, 110]
# mean_queue_size_updated = [274.7471264367816, 316.72701149425285, 375.20402298850576, 418.0749279538905, 474.04597701149424, 510.33045977011494]

# # Data for Mean Queue Size (Previous Prompt)
# mean_queue_size_previous = [1.9741379310344827, 44.195402298850574, 93.45402298850574, 115.84482758620689, 152.08333333333334, 196.51724137931035]

# # Plotting the data
# plt.figure(figsize=(14, 7))

# # Plot for Mean Queue Size (Updated Prompt)
# plt.subplot(1, 2, 1)
# plt.plot(requests_per_minute, mean_queue_size_updated, marker='o', linestyle='-', color='b', label='Updated Prompt')
# plt.title('Mean Queue Size (Updated Prompt)')
# plt.xlabel('Requests Per Minute')
# plt.ylabel('Mean Queue Size')
# plt.grid(True)
# plt.legend()

# # Plot for Mean Queue Size (Previous Prompt)
# plt.subplot(1, 2, 2)
# plt.plot(requests_per_minute, mean_queue_size_previous, marker='o', linestyle='-', color='r', label='Previous Prompt')
# plt.title('Mean Queue Size (Previous Prompt)')
# plt.xlabel('Requests Per Minute')
# plt.ylabel('Mean Queue Size')
# plt.grid(True)
# plt.legend()

# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt

# # Data for Mean Queue Waiting Time (Updated Prompt)
# requests_per_minute = [60, 70, 80, 90, 100, 110]
# mean_waiting_time_updated = [4.472018865673804, 11.775062493887154, 9.395630850054278, 9.003110482931222, 12.4303398561, 8.795335264321688]

# # Data for Mean Queue Waiting Time (Previous Prompt)
# mean_waiting_time_previous = [0.003669023211469663, 0.0815597951405056, 0.17370126732433375, 0.20559598413178454, 0.2644341765917634, 0.349346459707854]

# # Plotting the data
# plt.figure(figsize=(14, 7))

# # Plot for Mean Queue Waiting Time (Updated Prompt)
# plt.subplot(1, 2, 1)
# plt.plot(requests_per_minute, mean_waiting_time_updated, marker='o', linestyle='-', color='b', label='Updated Prompt')
# plt.title('Mean Queue Waiting Time (Updated Prompt)')
# plt.xlabel('Requests Per Minute')
# plt.ylabel('Mean Queue Waiting Time (seconds)')
# plt.grid(True)
# plt.legend()

# # Plot for Mean Queue Waiting Time (Previous Prompt)
# plt.subplot(1, 2, 2)
# plt.plot(requests_per_minute, mean_waiting_time_previous, marker='o', linestyle='-', color='r', label='Previous Prompt')
# plt.title('Mean Queue Waiting Time (Previous Prompt)')
# plt.xlabel('Requests Per Minute')
# plt.ylabel('Mean Queue Waiting Time (seconds)')
# plt.grid(True)
# plt.legend()

# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt

# Data points
mean_queue_size = [2.5689655172413794, 57.202739726027396, 112.48563218390805, 
                   173.98563218390805, 235.617816091954, 267.5214899713467, 
                   320.4051724137931, 493.5977011494253, 535.3218390804598]

mean_queue_time = [0.010700840787421038, 0.2434661774683751, 0.430338668579115, 
                   0.648335563350473, 0.7752293401205905, 0.8353501966973952, 
                   0.9456291099981107, 5.352308340166387, 3.900366620237936]

# Plotting Mean Queue Time against Mean Queue Size
plt.figure(figsize=(10, 6))
plt.plot(mean_queue_size, mean_queue_time, marker='o', linestyle='-', color='b')
plt.title('Mean Queue Time vs Mean Queue Size (MiG3g-LLama-Old Prompt)')
plt.xlabel('Mean Queue Size')
plt.ylabel('Mean Queue Time')
plt.grid(True)
plt.show()

# import matplotlib.pyplot as plt

# # Data points
# mean_queue_size = [35.87719298245614, 85.17241379310344, 133.19252873563218, 
#                    187.26724137931035, 229.31034482758622, 283.03735632183907, 
#                    326.04597701149424, 381.9568965517241, 435.53735632183907, 
#                    487.51149425287355, 545.8587896253603]

# mean_queue_time = [1.739838126489225, 6.303671556230154, 9.55463767306928, 
#                    5.880266256126801, 14.42369247345134, 27.1796722169421, 
#                    23.523210682086127, 24.688381273193286, 12.701925933527292, 
#                    11.200561169221196, 5.109288093819468]

# # Plotting Mean Queue Time against Mean Queue Size
# plt.figure(figsize=(10, 6))
# plt.plot(mean_queue_size, mean_queue_time, marker='o', linestyle='-', color='b')
# plt.title('Mean Queue Time vs Mean Queue Size (MiG3g.40gb-Llama-New Prompt)')
# plt.xlabel('Mean Queue Size')
# plt.ylabel('Mean Queue Time')
# plt.grid(True)
# plt.show()

