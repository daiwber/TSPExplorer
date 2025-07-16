import math
import itertools
import time
import random
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def ctd(f, p):
    res=0
    for i in range(len(f) - 1):
        res += dist(p[f[i]], p[f[i + 1]])
    res += dist(p[f[-1]], p[f[0]])
    #print(res)
    return res

def bf_tsp(p):
    n = len(p)
    mid = float('inf')
    bp = None
    for perm in itertools.permutations(range(n)):
        dist = ctd(perm, p)
        if dist < mid:
            mid = dist
            bp = perm
    return bp
def chrisalgo(points):
    n = len(points)
    dist_matrix = [[dist(points[i], points[j]) for j in range(n)] for i in range(n)]
    mst = minispantree(dist_matrix)
    odd_nodes = fodn(mst)
    if len(odd_nodes) < 11:
        odd_pairs = compute_perfect_matching(odd_nodes, dist_matrix)
    else:
        odd_pairs = compute_perfect_matching(odd_nodes, dist_matrix)
        #print(odd_pairs)
    ep = fep(mst, odd_pairs)
    hamipath = exhapa(ep)
    return hamipath
def minispantree(dist_matrix):
    n = len(dist_matrix)
    selected = [False] * n
    selected[0] = True
    mst_adj_matrix = [[0] * n for _ in range(n)]
    for _ in range(n - 1):
        min_edge = float('inf')
        a = b = -1
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if not selected[j] and dist_matrix[i][j]:
                        if dist_matrix[i][j] < min_edge:
                            min_edge = dist_matrix[i][j]
                            a, b = i, j
        if a != -1 and b != -1:
            mst_adj_matrix[a][b] = dist_matrix[a][b]
            mst_adj_matrix[b][a] = dist_matrix[a][b]
            selected[b] = True
    return mst_adj_matrix
def fodn(mst):
    odd_degree_nodes = []
    n = len(mst)
    for i in range(n):
        degree = sum(1 for j in range(n) if mst[i][j] != 0)
        if degree % 2 != 0:
            odd_degree_nodes.append(i)

    return odd_degree_nodes
def create_bipartite_graph(odd_nodes, dist_matrix):
    num_nodes = len(odd_nodes)
    cost_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cost_matrix[i][j] = dist_matrix[odd_nodes[i]][odd_nodes[j]]
            else:
                cost_matrix[i][j] = float('inf')
    return cost_matrix
def compute_perfect_matching(odd_nodes, dist_matrix):
    cost_matrix = create_bipartite_graph(odd_nodes, dist_matrix)
    assignment, min_cost = hun_algo(cost_matrix)
    best_matching = []
    for i, j in assignment.items():
        best_matching.append((odd_nodes[i], odd_nodes[j]))
    return rde(best_matching)
############################################################################################
######################################################################################################
def hun_algo(cost_matrix):
    n = len(cost_matrix)
    ocm = [row[:] for row in cost_matrix]
    for i in range(n):
        min_val = min(cost_matrix[i])
        for j in range(n):
            cost_matrix[i][j] -= min_val
    for j in range(n):
        min_val = min([cost_matrix[k][j] for k in range(n)])
        for i in range(n):
            cost_matrix[i][j] -= min_val
    star = [[False for _ in range(n)] for _ in range(n)]
    prime = [[False for _ in range(n)] for _ in range(n)]
    rows_covered = [False] * n
    cols_covered = [False] * n
    for i in range(n):
        for j in range(n):
            if cost_matrix[i][j] == 0 and not rows_covered[i] and not cols_covered[j]:
                star[i][j] = True
                rows_covered[i] = True
                cols_covered[j] = True
    while True:
        cover_lines = sum(rows_covered) + sum(cols_covered)
        if cover_lines >= n:
            break
        min_uncovered = float('inf')
        for i in range(n):
            for j in range(n):
                if not rows_covered[i] and not cols_covered[j] and cost_matrix[i][j] < min_uncovered:
                    min_uncovered = cost_matrix[i][j]
        for i in range(n):
            for j in range(n):
                if not rows_covered[i] and not cols_covered[j]:
                    cost_matrix[i][j] -= min_uncovered
                if rows_covered[i] and cols_covered[j]:
                    cost_matrix[i][j] += min_uncovered
        rows_covered = [False] * n
        cols_covered = [False] * n
        for i in range(n):
            for j in range(n):
                if cost_matrix[i][j] == 0 and not rows_covered[i] and not cols_covered[j]:
                    star[i][j] = True
                    rows_covered[i] = True
                    cols_covered[j] = True
    assignment = {}
    for i in range(n):
        for j in range(n):
            if star[i][j]:
                assignment[i] = j
    min_cost = 0
    for i, j in assignment.items():
        min_cost += ocm[i][j]
    return assignment, min_cost
def rde(bm):
    unique_edges = set()
    final_matching = []
    for pair in bm:
        unordered_pair = frozenset(pair)
        if unordered_pair not in unique_edges:
            unique_edges.add(unordered_pair)
            final_matching.append(pair)
    return final_matching
def fep(mst, odd_pairs):
    from collections import defaultdict
    graph = defaultdict(list)
    for i in range(len(mst)):
        for j in range(len(mst)):
            if mst[i][j] != 0:
                graph[i].append(j)
    for u, v in odd_pairs:
        graph[u].append(v)
        graph[v].append(u)
    def dfs(current):
        path = []
        stack = [current]
        while stack:
            u = stack[-1]
            if graph[u]:
                v = graph[u][-1]
                stack.append(v)
                graph[u].pop()
                if u in graph[v]:
                    graph[v].remove(u)
            else:
                path.append(stack.pop())
        return path
    start_node = odd_pairs[0][0] if odd_pairs else 0
    eulerian_path = dfs(start_node)
    return eulerian_path


def exhapa(eulerian_path):
    hamiltonian_path = []
    visited = set()
    for node in eulerian_path:
        if node not in visited:
            hamiltonian_path.append(node)
            visited.add(node)
    return hamiltonian_path

def clopof(group, selected_points):
    if not selected_points:
        return group[0]
    min_distance = float('inf')
    closest_point = None
    reference_point = selected_points[-1]
    for point in group:
        distance = dist(point, reference_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point

def calculate_centroid(group):
    tx = 0
    ty = 0
    n = 0
    for point in group:
        tx += point[0]
        ty += point[1]
        n += 1
    centroid = (tx / n, ty / n)
    return centroid

def fssp(group, centroid):
    min_distance = float('inf')
    closest_point = None
    for point in group:
        distance = caldist_dai(centroid, point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point

def sfpv1(group, selected_points):
    max_min_distance = 0
    best_point = group[0]

    for point in group:
        mid = float('inf')
        for selected_point in selected_points:
            distance = caldist_dai(point, selected_point)
            if distance < mid:
                mid = distance
        if mid > max_min_distance:
            max_min_distance = mid
            best_point = point

    return best_point

def sfqq_chen(group):
    min_distance = float('inf')
    closest_point = None

    for point in group:
        distance = caldist_dai(point, (0, 0))
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point

def fpsssk(ring, selected_points):
    if not selected_points:
        return ring[0]
    max_distance = 0
    farthest_point = None
    current_point = selected_points[0]
    for point in ring:
        distance = dist(current_point, point)
        if distance > max_distance:
            max_distance = distance
            farthest_point = point
    return farthest_point

def ccen(group):
    total_x = sum(point[0] for point in group)
    total_y = sum(point[1] for point in group)
    n = len(group)
    return (total_x / n, total_y / n)

def caldist_dai(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def fclp(group, center):
    min_distance = float('inf')
    closest_point = None
    for point in group:
        distance = caldist_dai(point, center)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point

def sol(points):
    shortest_path = None
    shortest_distance = float('inf')
    selection_strategies = [
        "closest_to_centroid",
        "farthest_point_sampling",
        "closest_point_first",
        "feature_point_v1",
        "feature_point_v2",
        "random_selection",
    ]
    centroids = [ccen(group) for group in points if group]
    center = ccen(centroids)

    for strategy in selection_strategies:
        selected_points = []
        for group in points:
            if group:
                if strategy == "closest_to_centroid":
                    selected_point = fclp(group, center)
                elif strategy == "farthest_point_sampling":
                    selected_point = fpsssk(group, selected_points)
                elif strategy == "closest_point_first":
                    selected_point = clopof(group, selected_points)
                elif strategy == "feature_point_v1":
                    selected_point = sfpv1(group, selected_points)
                elif strategy == "feature_point_v2":
                    selected_point = sfqq_chen(group)
                elif strategy == "random_selection":
                    selected_point = random.choice(group)  # 随机选择一个点
                selected_points.append(selected_point)  # 添加选择的点

        if len(selected_points) < 10:
            best_path = bf_tsp(selected_points)
        else:
            best_path = chrisalgo(selected_points)

        path_distance = ctd(best_path, selected_points)

        if path_distance < shortest_distance:
            shortest_distance = path_distance
            shortest_path = [selected_points[i] for i in best_path]
    timeout = 3
    start_time = time.time()

    cnt = 0
    while True:
        cnt += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            #print(cnt)
            #print("达到超时限制，停止随机选择。")
            break

        random_selected_points = [random.choice(group) for group in points if group]

        if len(random_selected_points) < 10:
            best_path = bf_tsp(random_selected_points)
        else:
            best_path = chrisalgo(random_selected_points)

        path_distance = ctd(best_path, random_selected_points)

        if path_distance < shortest_distance:
            shortest_distance = path_distance
            shortest_path = [random_selected_points[i] for i in best_path]
            # print("随机化优化：")
            # print("最短路径:", shortest_path)
            # print("最短路径长度:", shortest_distance)

    # print("最短路径:", shortest_path)
    # print("最短路径长度:", shortest_distance)

    return shortest_path

def parse_input():
    input_str = input().strip()
    parts = input_str.split('@')
    points = []
    for part in parts:
        part = part.strip()
        if part.startswith('[') and part.endswith(']'):
            point_list = part[2:-2].split('), (')
            points_group = [tuple(map(int, point.split(','))) for point in point_list]
            points.append(points_group)
    return points

def main():
    # 1. 记录开始时间
    start_time = time.time()

    points = parse_input()
    result = sol(points)
    print(result)

    #
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time:.6f} 秒")

if __name__ == '__main__':
    main()

# =========
# 8.17分 / 10
# 8.92分 / 10
# 31.74分 / 50
# 16.72分 / 50
# 36.27分 / 50
# 57.41分 / 75
# 49.13分 / 75
# 139.80分 / 150
# 127.00分 / 150
# 261.33分  / 300
####第一次得分

######################

# 16.65分
# 15.88分
# 32.90分
# 18.43分
# 38.15分
# 57.41分
# 编译或运行超时
# 146.03分
# 144.45分
# 265.80分
##################################