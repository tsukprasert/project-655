import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

np.random.seed(42)
trace_file = "/Users/tsukprasert/Library/CloudStorage/OneDrive-UniversityofMassachusetts/Desktop/project-655-v2/transform_trace/mm1_trace/azure.csv"
trace_df = pd.read_csv(trace_file)
total_arrival_rate = 1 / trace_df['arrival_time'].diff().mean()

mu = 11
total_simulation_time = 5000
class MM1Queue:
    def __init__(self, service_rate, id, max_cpu):
        self.service_rate = service_rate
        self.id = id
        self.queue = []  # Tracks jobs (arrival times, job length, required CPUs)
        self.events = []  # Tracks all events (arrival, departure)
        self.num_in_queue_over_time = []  # Tracks number of jobs in the queue over time
        self.total_simulation_time = 0
        self.max_cpu = max_cpu
        self.current_cpu_usage = 0  # Track current CPU usage
        self.dropped_jobs = 0  # Track number of dropped jobs

        # Metrics tracking
        self.num_arrivals = 0
        self.num_departures = 0
        self.total_time_in_system = 0
        self.total_time_in_queue = 0

    def add_job(self, arrival_time, job_length, required_cpus, index):
        """Add a new job to the queue."""
        # Check if there are enough CPUs available
        if required_cpus > self.max_cpu:
            # print(f"Job {index} requires more CPUs than available. Dropping job.")
            self.dropped_jobs += 1
            return False

        self.queue.append((arrival_time, job_length, required_cpus, index))
        self.events.append((arrival_time, "Arrival", self.id))
        self.num_in_queue_over_time.append((arrival_time, len(self.queue)))
        self.num_arrivals += 1
        return True

    def simulate(self, max_time):
        """Simulate the processing of jobs in the queue."""
        current_time = 0
        self.total_simulation_time = max_time
        customers_in_system = []  # Track customers in system at each time step

        while self.queue and current_time < max_time:
            # Get the next job
            arrival_time, job_length, required_cpus, index = self.queue.pop(0)

            # Check if there are enough CPUs available
            if self.current_cpu_usage + required_cpus > self.max_cpu:
                print(f"Job {index} cannot be processed due to CPU constraints. Dropping job.")
                self.dropped_jobs += 1
                self.num_in_queue_over_time.append((current_time, len(self.queue)))
                continue

            # Update CPU usage
            self.current_cpu_usage += required_cpus
            # print(self.current_cpu_usage, required_cpus)

            # Wait time in queue
            wait_time_in_queue = max(0, current_time - arrival_time)
            self.total_time_in_queue += wait_time_in_queue

            # Service time is the job length
            service_time = job_length
   
            current_time = max(current_time, arrival_time) + service_time

            # Total time in system
            time_in_system = current_time - arrival_time
            self.total_time_in_system += time_in_system

            # Release CPUs
            self.current_cpu_usage -= required_cpus

            # Record departure
            self.events.append((current_time, "Departure", self.id))
            self.num_in_queue_over_time.append((current_time, len(self.queue)))
            self.num_departures += 1

            # Track the number of customers in the system at this time
            customers_in_system.append((current_time, len(self.queue) + 1))  # +1 for the job being processed

        # Store the number of customers in the system
        self.customers_in_system = customers_in_system

    def calculate_metrics(self, arrival_rate):


        """Calculate and print the metrics for the queue."""
        # Compute utilization
        avg_job_length = self.total_time_in_system / max(1, self.num_departures) if self.num_departures > 0 else 0
        rho = avg_job_length * arrival_rate

        # Empirical metrics
        avg_num_customers_in_system = self.total_time_in_system / self.total_simulation_time
        avg_num_customers_in_queue = self.total_time_in_queue / self.total_simulation_time


        # return self.num_arrivals, self.num_departures, self.dropped_jobs
        print(f"Queue {self.id} Metrics:")
        print(f"Empirical average number of customers in the system: {avg_num_customers_in_system:.4f}")
        print(f"Empirical average number of customers in the queue: {avg_num_customers_in_queue:.4f}")
        print(f"Utilization (ρ): {rho:.4f}")
        print(f"Number of arrivals: {self.num_arrivals}")
        print(f"Number of departures: {self.num_departures}")
        print(f"Number of dropped jobs: {self.dropped_jobs}\n")

class LoadBalancer:
    def __init__(self, queues, policy="round_robin", n_random=2):
        self.queues = queues
        self.policy = policy
        self.current_queue_index = 0
        self.n_random = n_random
        self.total_dropped_jobs = 0

    def select_queue(self):

        eligible_queue = [queue for queue in self.queues if len(queue.queue) < queue.max_cpu]
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

    def simulate(self, max_time, total_arrival_rate, job_trace_file):
        # Read job trace file
        job_trace_df = pd.read_csv(job_trace_file)
        
        current_time = 0
        self.total_simulation_time = max_time

        for index, row in job_trace_df.iterrows():
            arrival_time = row['arrival_time']
            job_length = row['length']
            required_cpus = row['cpus']

            if arrival_time > max_time:
                break

            # Find the next selected queue
            selected_queue = self.select_queue()

            # Attempt to add job to the queue
            job_added = selected_queue.add_job(arrival_time, job_length, required_cpus, index)
            
            if not job_added:
                self.total_dropped_jobs += 1

        # Simulate each queue
        for queue in self.queues:
            queue.simulate(max_time)

    def display_metrics(self, total_arrival_rate):
        per_queue_arrival_rate = total_arrival_rate / len(self.queues)
        print(f"Total dropped jobs across all queues: {self.total_dropped_jobs}\n")

        arrival_df = pd.DataFrame()
        for queue in self.queues:
            queue.calculate_metrics(per_queue_arrival_rate)

    def save_stats(self, savetodir):
        # policy_dir = f"{savetodir}/{self.policy}"
        queue_df = pd.DataFrame(columns=['arrival', 'departure', 'drop'])

        for queue in self.queues:
            arrival = queue.num_arrivals
            departure = queue.num_departures
            drop = queue.dropped_jobs
            queue_df = queue_df._append({'arrival': arrival, 'departure': departure, 'drop': drop}, ignore_index=True)
        
        queue_df.to_csv(f"{savetodir}/{self.policy}.csv", index=False)

def main(policy, savetodir, job_trace_file, total_simulation_time, total_arrival_rate):
    # Create queues with different service rates and CPU capacities
    queue1 = MM1Queue(service_rate=7, id=1, max_cpu=1)  # Slow Server
    queue2 = MM1Queue(service_rate=10, id=2, max_cpu=2)  # Moderate Server
    queue3 = MM1Queue(service_rate=16, id=3, max_cpu=3)  # Fast Server

    load_balancer = LoadBalancer(
        queues=[queue1, queue2, queue3],
        policy=policy
    )

    load_balancer.simulate(
        max_time=total_simulation_time, 
        total_arrival_rate=total_arrival_rate,
        job_trace_file=job_trace_file
    )

    load_balancer.save_stats(savetodir)
    # load_balancer.display_metrics(total_arrival_rate)

if __name__ == "__main__":
    # Job trace file
    trace_dir = "/Users/tsukprasert/Library/CloudStorage/OneDrive-UniversityofMassachusetts/Desktop/project-655-v2/transform_trace/mm1_trace_v2"
    
    files = os.listdir(trace_dir)


    savetodir = 'data/heterogeneous_cpu_v3'
    if not os.path.exists(savetodir):
        os.mkdir(savetodir)

    for file in files:
        
        source = file.replace('.csv', '') 

        trace_file = os.path.join(trace_dir, file)

        trace_df = pd.read_csv(trace_file)
        total_arrival_rate = 1 / trace_df['arrival_time'].diff().mean()
        total_simulation_time = 1000

        policies = [
            "round_robin", 
            "least_loaded", 
            "random", 
            "random_n_least_loaded"
        ]


        sourcedir = os.path.join(savetodir, source)
        if not os.path.exists(sourcedir):
            os.mkdir(sourcedir) 

        for policy in policies:
            main(policy, sourcedir, trace_file, total_simulation_time, total_arrival_rate)

