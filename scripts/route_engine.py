from collections import deque


def find_path_bfs(start_node, end_node, stop_to_routes, route_to_stops):
    queue = deque([(start_node, [])])
    visited_stops = {start_node}

    while queue:
        current_stop, path = queue.popleft()

        if current_stop == end_node:
            return path

        for bus_id in stop_to_routes.get(str(current_stop), []):

            for next_stop in route_to_stops.get(bus_id, []):
                next_stop_str = str(next_stop)

                if next_stop_str not in visited_stops:
                    visited_stops.add(next_stop_str)


                    new_path = path + [{"bus": bus_id, "get_off_at": next_stop_str}]
                    queue.append((next_stop_str, new_path))

                    if next_stop_str == end_node:
                        return new_path
    return None