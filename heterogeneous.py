import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

np.random.seed(42)


trace_file = "/Users/tsukprasert/Library/CloudStorage/OneDrive-UniversityofMassachusetts/Desktop/project-655-v2/transform_trace/mm1_trace_v2/azure.csv"
trace_df = pd.read_csv(trace_file)
total_arrival_rate = 1 / trace_df['arrival_time'].diff().mean()

mu = 11
total_simulation_time = 5000

class MM1Queue:
    def __init__(self, service_rate, id):
        self.service_rate = service_rate
        self.id = id
        self.queue = []  # Tracks jobs (arrival times)
        self.events = []  # Tracks all events (arrival, departure)
        self.num_in_queue_over_time = []  # Tracks number of jobs in the queue over time
        self.total_simulation_time = 0

        # Metrics tracking
        self.num_arrivals = 0
        self.num_departures = 0
        self.total_time_in_system = 0
        self.total_time_in_queue = 0

    def add_job(self, arrival_time, index):
        """Add a new job to the queue."""
        self.queue.append((arrival_time, index))
        self.events.append((arrival_time, "Arrival", self.id))
        self.num_in_queue_over_time.append((arrival_time, len(self.queue)))
        self.num_arrivals += 1

    def simulate(self, max_time):
        """Simulate the processing of jobs in the queue."""
        current_time = 0
        self.total_simulation_time = max_time
        customers_in_system = []  # Track customers in system at each time step


        while self.queue and current_time < max_time:
            # Get the next job
            arrival_time, index = self.queue.pop(0)

            # Wait time in queue
            wait_time_in_queue = max(0, current_time - arrival_time)
            self.total_time_in_queue += wait_time_in_queue

            # Service time
            service_time = np.random.exponential(1 / self.service_rate)
   
            current_time = max(current_time, arrival_time) + service_time

            # Total time in system
            time_in_system = current_time - arrival_time
            self.total_time_in_system += time_in_system

            # Record departure
            self.events.append((current_time, "Departure", self.id))
            self.num_in_queue_over_time.append((current_time, len(self.queue)))
            self.num_departures += 1

            # Track the number of customers in the system at this time
            customers_in_system.append((current_time, len(self.queue) + 1))  # +1 for the job being processed

        # Store the number of customers in the system
        self.customers_in_system = customers_in_system

    def calculate_metrics(self, arrival_rate):
        """Calculate and print the average number of jobs in the system and queue."""
        # Compute utilization
        rho = arrival_rate / self.service_rate

        # Empirical metrics
        avg_num_customers_in_system = self.total_time_in_system / self.total_simulation_time
        avg_num_customers_in_queue = self.total_time_in_queue / self.total_simulation_time

        # Theoretical metrics
        theoretical_l = rho / (1 - rho)  # Average number in system
        theoretical_lq = rho ** 2 / (1 - rho)  # Average number in queue


        print(f"Queue {self.id} Metrics:")
        print(f"Empirical average number of customers in the system: {avg_num_customers_in_system:.4f}")
        print(f"Theoretical average number of customers in the system: {theoretical_l:.4f}")
        print(f"Empirical average number of customers in the queue: {avg_num_customers_in_queue:.4f}")
        print(f"Theoretical average number of customers in the queue: {theoretical_lq:.4f}")
        print(f"Utilization (ρ): {rho:.4f}")
        print(f"Number of arrivals: {self.num_arrivals}")
        print(f"Number of departures: {self.num_departures}\n")


class LoadBalancer:
    def __init__(self, queues, policy="round_robin", n_random=2):
        self.queues = queues
        self.policy = policy
        self.current_queue_index = 0
        self.n_random = n_random

    def select_queue(self):
        if self.policy == "round_robin":
            selected_queue = self.queues[self.current_queue_index]
            self.current_queue_index = (self.current_queue_index + 1) % len(self.queues)

        elif self.policy == "least_loaded":
            selected_queue = min(self.queues, key=lambda q: len(q.queue) / q.service_rate)


        elif self.policy == "random":
            selected_queue = np.random.choice(self.queues)

        elif self.policy == "random_n_least_loaded":
            random_queues = np.random.choice(self.queues, size=min(self.n_random, len(self.queues)), replace=False)
            selected_queue = min(random_queues, key=lambda q: len(q.queue) / q.service_rate)

        else:
            raise ValueError(f"Unsupported policy: {self.policy}")

        return selected_queue

    def simulate(self, max_time, total_arrival_rate):
        current_time = 0
        self.total_simulation_time = max_time

        index = 0

        while current_time < max_time:
            interarrival_time = np.random.exponential(1 / total_arrival_rate)
            current_time += interarrival_time

            if current_time <= max_time:
                selected_queue = self.select_queue()
                selected_queue.add_job(current_time, index)
                index += 1

        for queue in self.queues:
            queue.simulate(max_time)

    def display_metrics(self, total_arrival_rate):
        per_queue_arrival_rate = total_arrival_rate / len(self.queues)
        for queue in self.queues:
            queue.calculate_metrics(per_queue_arrival_rate)


    def save_queue_events(self, savetodir):

        policy_dir = f"{savetodir}/{self.policy}"
        if not os.path.exists(policy_dir): 
            os.mkdir(policy_dir)


        # Initialize empty DataFrames
        arrival_df = pd.DataFrame()
        departure_df = pd.DataFrame()

        for queue in self.queues:
            # Extract arrival and departure times
            arrival_times = [event[0] for event in queue.events if event[1] == "Arrival"]
            departure_times = [event[0] for event in queue.events if event[1] == "Departure"]

            # Determine the current maximum length of the DataFrames
            max_length = max(len(arrival_df), len(departure_df), len(arrival_times), len(departure_times))

            # Pad the current DataFrames with None if their length is less than max_length
            while len(arrival_df) < max_length:
                arrival_df = arrival_df._append(pd.Series(), ignore_index=True)
            while len(departure_df) < max_length:
                departure_df = departure_df._append(pd.Series(), ignore_index=True)

            # Pad the new lists with None if their length is less than max_length
            while len(arrival_times) < max_length:
                arrival_times.append(None)
            while len(departure_times) < max_length:
                departure_times.append(None)

            # Add the lists as columns to the DataFrames
            arrival_df[queue.id] = arrival_times
            departure_df[queue.id] = departure_times


            arrival_df.to_csv(f"{policy_dir}/arrival.csv", index=False)
            departure_df.to_csv(f"{policy_dir}/departure.csv", index=False)

        print("Data saved to", policy_dir)

    def plot_queue_events(self):
        plt.figure(figsize=(12, 8))

        # Plot all arrivals in one histogram
        plt.subplot(2, 1, 1)  # Top subplot for arrivals
        for queue in self.queues:
            arrival_times = [event[0] for event in queue.events if event[1] == "Arrival"]
            # Plot each queue's arrivals as a line histogram (no shading)
            plt.hist(arrival_times, bins=100, alpha=0.7, histtype='step', label=f"Queue {queue.id} Arrivals", linewidth=2)
        
        plt.xlabel("Time")
        plt.ylabel("Number of Arrivals")
        plt.title("Arrival Events for Each Queue")
        plt.legend()
        plt.grid(True)

        # Plot all departures in one histogram
        plt.subplot(2, 1, 2)  # Bottom subplot for departures
        for queue in self.queues:
            departure_times = [event[0] for event in queue.events if event[1] == "Departure"]
            # Plot each queue's departures as a line histogram (no shading)
            plt.hist(departure_times, bins=100, alpha=0.7, histtype='step', label=f"Queue {queue.id} Departures", linewidth=2)
        
        plt.xlabel("Time")
        plt.ylabel("Number of Departures")
        plt.title("Departure Events for Each Queue")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()



    def plot_total_customers_in_system(self):
        plt.figure(figsize=(12, 8))

        # Plot the number of customers in the system for each queue
        for queue in self.queues:
            times = [t[0] for t in queue.customers_in_system]
            num_customers = [t[1] for t in queue.customers_in_system]
            
            plt.plot(times, num_customers, label=f"Queue {queue.id}")

        plt.xlabel("Time")
        plt.ylabel("Number of Customers in the System")
        plt.title("Total Number of Customers in the System for Each Queue")
        plt.legend()
        plt.grid()
        plt.show()


def main(policy, savetodir):

    queue1 = MM1Queue(service_rate=7, id=1) # Slow Server
    queue2 = MM1Queue(service_rate=10, id=2) # Moderate Server
    queue3 = MM1Queue(service_rate=16, id=3) # Fast Server

    load_balancer = LoadBalancer(
        
        queues=[queue1, queue2, queue3],
        policy=policy
    )

    load_balancer.simulate(max_time=total_simulation_time, total_arrival_rate=total_arrival_rate)
    load_balancer.save_queue_events(savetodir)
    load_balancer.display_metrics(total_arrival_rate)
    # load_balancer.plot_queue_events()

if __name__ == "__main__":

    policies = [
        "round_robin", 
        "least_loaded", 
        "random", 
        "random_n_least_loaded"
    ]

    savetodir = 'data/heterogeneous_v2'
    if not os.path.exists(savetodir):
        os.mkdir(savetodir)

    for policy in policies:
        main(policy, savetodir)
