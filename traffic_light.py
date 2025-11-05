def calculate_weighted_volume(vehicle_counts, weights):
   
    return sum(vehicle_counts[vehicle] * weights[vehicle] for vehicle in vehicle_counts)

def allocate_traffic_light_times(road_volumes, total_cycle_time):
   
    road_times = {}

    # Calculate green time directly as volume * x
    for road, volume in road_volumes.items():
        green_time = volume * 0.15 # Calculate green light time
        road_times[road] = {"green": green_time, "red": total_cycle_time - green_time}

    return road_times
