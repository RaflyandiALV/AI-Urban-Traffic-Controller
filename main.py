import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Gunakan backend Tkinter untuk pop-up GUI
from process_vehicles import detect_objects_for_all_remaining_roads, VEHICLE_CLASSES
from traffic_light import calculate_weighted_volume, allocate_traffic_light_times

# Define weights for vehicles
weights = {"motorcycle": 1, "car": 3, "bus": 6, "truck": 6}

# Parameters
total_cycle_time = 120  # Total traffic light cycle time in seconds
yellow_light_duration = 5  # Yellow light duration in seconds
scan_index = 1  # Start with the first image (e.g., image1.jpg)


def visualize_traffic(road_volumes, current_light, priority_road=None):
   
    roads = list(road_volumes.keys())
    volumes = list(road_volumes.values())

    plt.bar(roads, volumes, color='blue', alpha=0.7)
    plt.title(f"Traffic Light: {current_light} | Priority Road: {priority_road or 'None'}")
    plt.xlabel("Roads")
    plt.ylabel("Traffic Volume")
    plt.ylim(0, max(volumes) + 10 if volumes else 10)
    plt.grid(axis='y')
    plt.pause(0.5)
    plt.clf()

def visualize_pie_chart(road_volumes):
   
    labels = road_volumes.keys()
    sizes = road_volumes.values()

    # Check if all sizes are zero
    if sum(sizes) == 0:
        print("No traffic detected. Skipping pie chart visualization.")
        plt.figure()
        plt.title("No traffic data to visualize")
        plt.text(0.5, 0.5, "No Data", fontsize=20, ha='center')
        plt.axis("off")
        plt.show(block=False)
        plt.pause(1)
        plt.clf()
        return

    # Generate pie chart if sizes are non-zero
    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Traffic Volume Distribution")
    plt.show(block=False)
    plt.pause(1)
    plt.clf()


def visualize_line_chart(road_volumes_history, all_roads):
    
    plt.figure()
    for road in all_roads:
        # Ensure road exists in all histories with default value 0
        history = [cycle.get(road, 0) for cycle in road_volumes_history]
        plt.plot(range(len(history)), history, label=road)

    plt.title("Traffic Volume Over Time")
    plt.xlabel("Cycle")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.grid()
    plt.show(block=False)
    plt.pause(1)
    plt.clf()


plt.ion()  # Turn on interactive mode for live updates

roads_to_process = ["road_1", "road_2", "road_3", "road_4"]  # List of roads
road_volumes_history = []

while True:
    # Detect objects for all roads and calculate volumes
    print("Detecting objects for all roads...")
    detected_counts = detect_objects_for_all_remaining_roads(roads_to_process, scan_index)
    print(f"Vehicle Counts: {detected_counts}")

    # Ensure road_volumes includes all roads with default 0
    road_volumes = {
    road: calculate_weighted_volume(detected_counts.get(road, {cls: 0 for cls in VEHICLE_CLASSES}), weights)
    for road in roads_to_process
    }

    road_volumes_history.append(road_volumes)
    print(f"Road Volumes: {road_volumes}")

    visualize_pie_chart(road_volumes)

    while roads_to_process:
        # Handle case when all road volumes are zero
        if sum(road_volumes.values()) == 0:
            print("All roads have zero volume. Setting all lights to Yellow...")
            for road in roads_to_process:
                print(f"[{road}] Yellow Light ON for {total_cycle_time} seconds...")
                visualize_traffic(road_volumes, "Yellow", road)
                time.sleep(total_cycle_time)
            break

        # Determine the priority road
        priority_road = max(road_volumes, key=road_volumes.get)
        print(f"Priority Road: {priority_road}")

        # Simulate yellow light phase
        print(f"\n[{priority_road}] Yellow Light ON for {yellow_light_duration} seconds...")
        visualize_traffic(road_volumes, "Yellow", priority_road)
        time.sleep(yellow_light_duration)

        # Calculate green light time based on traffic light times
        traffic_light_times = allocate_traffic_light_times(road_volumes, total_cycle_time)
        green_time = traffic_light_times[priority_road]["green"]
        print(f"[{priority_road}] Green Light ON for {green_time} seconds...")

        # Simulate green light
        visualize_traffic(road_volumes, "Green", priority_road)
        time.sleep(green_time - 15)  # Run for the first part of green light

        print(f"[{priority_road}] 15 seconds left for Green Light...")
        visualize_traffic(road_volumes, "Green", priority_road)
        time.sleep(15)

        # Simulate red light
        print(f"[{priority_road}] Red Light ON")
        visualize_traffic(road_volumes, "Red", priority_road)

        # Remove the processed road
        roads_to_process.remove(priority_road)
        del road_volumes[priority_road]

        # Recalculate priorities for remaining roads
        if roads_to_process:
            print("\nRecalculating priorities for remaining roads...")
            road_volumes = {
                road: calculate_weighted_volume(detected_counts[road], weights)
                for road in roads_to_process
            }
            print(f"Road Volumes: {road_volumes}")

    visualize_line_chart(road_volumes_history, ["road_1", "road_2", "road_3", "road_4"])


    # Restart the cycle with all roads after the last road is processed
    print("Restarting cycle with all roads...\n")
    roads_to_process = ["road_1", "road_2", "road_3", "road_4"]
    scan_index += 1  # Move to the next image for detection (e.g., image2.jpg)
    plt.pause(2)  # Pause briefly before restarting

plt.ioff()
plt.show(block=True)

