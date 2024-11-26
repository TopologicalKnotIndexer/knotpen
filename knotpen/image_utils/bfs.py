import numpy as np
from scipy.ndimage import label
from collections import deque

def bfs(start, labeled_image):
    """ 使用 BFS 计算从起始点出发的最远距离、最远点和最远点对 """
    queue = deque([start])
    visited = set([start])
    max_distance = 0
    farthest_point = start

    # 存储每个点的距离
    distance_map = {start: 0}

    while queue:
        current = queue.popleft()
        current_distance = distance_map[current]

        # 检查八个方向
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if (0 <= nx < labeled_image.shape[0] and
                    0 <= ny < labeled_image.shape[1] and
                    (nx, ny) not in visited and
                    labeled_image[nx, ny] == labeled_image[start[0], start[1]]):
                visited.add((nx, ny))
                queue.append((nx, ny))
                distance_map[(nx, ny)] = current_distance + 1
                
                # 更新最远点
                if distance_map[(nx, ny)] > max_distance:
                    max_distance = distance_map[(nx, ny)]
                    farthest_point = (nx, ny)

    return max_distance, farthest_point

def get_diameter(labeled, num_features):
    results = []
    for feature in range(1, num_features + 1):
        coords = np.argwhere(labeled == feature)
        if len(coords) > 0:
            start_point = tuple(coords[0])                                # get first start point
            _, farthest_point = bfs(start_point, labeled)                 # first bfs
            diameter, other_farthest_point = bfs(farthest_point, labeled) # second bfs
            results.append((diameter, farthest_point, other_farthest_point))
    return results