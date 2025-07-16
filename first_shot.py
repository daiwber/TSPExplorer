import math
import itertools
import time


def euclidean_distance(p1, p2):
    """计算两个点之间的欧几里得距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def compute_total_distance(path, points):
    """计算路径的总距离"""
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += euclidean_distance(points[path[i]], points[path[i + 1]])
    total_distance += euclidean_distance(points[path[-1]], points[path[0]])  # 回到起点
    return total_distance

def brute_force_tsp(points):
    """暴力法解TSP问题：计算所有排列并选出最短路径"""
    n = len(points)
    min_distance = float('inf')
    best_path = None

    for perm in itertools.permutations(range(n)):
        # 计算每个排列的路径总距离
        dist = compute_total_distance(perm, points)
        if dist < min_distance:
            min_distance = dist
            best_path = perm

    return best_path


def christofides_algorithm(points):
    """使用Christofides算法进行TSP求解的近似"""
    n = len(points)
    # 1. 找到所有点的距离矩阵
    dist_matrix = [[euclidean_distance(points[i], points[j]) for j in range(n)] for i in range(n)]
    mst = minimum_spanning_tree(dist_matrix)
    odd_nodes = find_odd_degree_nodes(mst)
    if len(odd_nodes) < 11:
        odd_pairs = compute_perfect_matching_brute_force(odd_nodes, dist_matrix)
    else:
        odd_pairs = compute_perfect_matching_greedy(odd_nodes, dist_matrix)
    eulerian_path = find_eulerian_path(mst, odd_pairs)
    hamiltonian_path = extract_hamiltonian_path(eulerian_path)
    return hamiltonian_path


def minimum_spanning_tree(dist_matrix):
    """计算最小生成树，使用Prim算法实现"""
    n = len(dist_matrix)
    selected = [False] * n  # 记录哪些点已被加入到生成树
    selected[0] = True  # 从第一个节点开始
    mst_adj_matrix = [[0] * n for _ in range(n)]  # 初始化MST的邻接矩阵

    # 边的个数等于节点数减一
    for _ in range(n - 1):
        min_edge = float('inf')
        a = b = -1

        # 查找未选中节点中，连接已选中节点的最小边
        for i in range(n):
            if selected[i]:  # 如果节点已被选中
                for j in range(n):
                    if not selected[j] and dist_matrix[i][j]:  # 连接的边
                        if dist_matrix[i][j] < min_edge:
                            min_edge = dist_matrix[i][j]
                            a, b = i, j

                            # 记录选中的边到邻接矩阵
        if a != -1 and b != -1:  # 确保找到有效的边
            mst_adj_matrix[a][b] = dist_matrix[a][b]
            mst_adj_matrix[b][a] = dist_matrix[a][b]  # 因为是无向图
            selected[b] = True  # 将选中的节点标记为已选
    return mst_adj_matrix  # 返回MST的邻接矩阵


def find_odd_degree_nodes(mst):
    """查找具有奇数度的节点"""
    odd_degree_nodes = []

    n = len(mst)  # 获取节点的数量
    for i in range(n):
        # 计算第i个节点的度，度为与其相连的边的数量
        degree = sum(1 for j in range(n) if mst[i][j] != 0)  # 计算节点i的度
        if degree % 2 != 0:  # 检查度是否为奇数
            odd_degree_nodes.append(i)  # 添加到结果列表

    return odd_degree_nodes


############################################################################################
def calculate_matching_weight(matching, dist_matrix):
    """计算一组匹配的总权重"""
    total_weight = 0
    for u, v in matching:
        total_weight += dist_matrix[u][v]
    return total_weight

def backtrack(matched, odd_nodes, dist_matrix):
    """回溯法，生成所有可能的匹配组合"""
    if len(matched) == len(odd_nodes):
        return [matched]

    results = []
    for i in range(len(odd_nodes)):
        if odd_nodes[i] not in matched:
            for j in range(i + 1, len(odd_nodes)):
                if odd_nodes[j] not in matched:
                    # 尝试匹配odd_nodes[i] 和 odd_nodes[j]
                    new_matched = matched + [odd_nodes[i], odd_nodes[j]]
                    results += backtrack(new_matched, odd_nodes, dist_matrix)

    return results


def compute_perfect_matching_brute_force(odd_nodes, dist_matrix):
    """计算奇数度节点的完美匹配"""
    # 确保odd_nodes的数量为偶数
    if len(odd_nodes) % 2 != 0:
        raise ValueError("Odd nodes count must be even for perfect matching")

    all_matchings = backtrack([], odd_nodes, dist_matrix)

    # 计算每个匹配的权重，并找到最小值
    min_weight = float('inf')
    best_matching = None
    for matching in all_matchings:
        weight = calculate_matching_weight([(matching[i], matching[i + 1]) for i in range(0, len(matching), 2)],
                                           dist_matrix)
        if weight < min_weight:
            min_weight = weight
            best_matching = [(matching[i], matching[i + 1]) for i in range(0, len(matching), 2)]

    return best_matching
def compute_perfect_matching_greedy(odd_nodes, dist_matrix):
    matching = []
    odd_nodes_copy = odd_nodes[:]
    while odd_nodes_copy:
        min_distance = float('inf')
        min_pair = (-1, -1)
        for i in range(len(odd_nodes_copy)):
            for j in range(i + 1, len(odd_nodes_copy)):
                distance = dist_matrix[odd_nodes_copy[i]][odd_nodes_copy[j]]
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (odd_nodes_copy[i], odd_nodes_copy[j])
        matching.append(min_pair)
        odd_nodes_copy.remove(min_pair[0])
        odd_nodes_copy.remove(min_pair[1])
    return matching

def find_eulerian_path(mst, odd_pairs):
    """合并生成的匹配并构造欧拉路径"""
    from collections import defaultdict

    # Step 1: 构建一个完整图
    graph = defaultdict(list)

    # 将MST的边添加到图中
    for i in range(len(mst)):
        for j in range(len(mst)):
            if mst[i][j] != 0:  # 有边的情况
                graph[i].append(j)  # 添加边 i -> j

    # Step 2: 添加奇数度节点的匹配边
    for u, v in odd_pairs:
        graph[u].append(v)
        graph[v].append(u)

        # Step 3: 进行DFS以构建欧拉路径

    def dfs(current):
        path = []
        stack = [current]

        while stack:
            u = stack[-1]
            if graph[u]:  # 如果还有邻接边
                v = graph[u][-1]  # 查看最后一个连接的边
                stack.append(v)  # 向下遍历
                graph[u].pop()  # 移除该边

                # 移除在v的逆边，只有在v不为空时才进行
                if u in graph[v]:
                    graph[v].remove(u)
            else:
                path.append(stack.pop())  # 回溯并记录路径

        return path

        # 找到任意奇数度节点作为起点（假设odd_pairs不为空）

    start_node = odd_pairs[0][0] if odd_pairs else 0  # 如果没有奇数度节点，选择0为起点
    eulerian_path = dfs(start_node)

    return eulerian_path


def extract_hamiltonian_path(eulerian_path):
    """从欧拉路径中提取哈密尔顿路径"""
    hamiltonian_path = []
    visited = set()  # 用于检查节点是否已经被加入哈密尔顿路径

    for node in eulerian_path:
        if node not in visited:  # 如果未访问过该节点
            hamiltonian_path.append(node)
            visited.add(node)  # 标记为已访问

    return hamiltonian_path

def closest_point_first(group, selected_points):
    """选择与已选择点集中最近的点"""
    if not selected_points:
        return group[0]  # 如果还没有选择任何点，返回第一个点
    min_distance = float('inf')
    closest_point = None
    reference_point = selected_points[-1]  # 参考最后一个已选择的点
    for point in group:
        distance = euclidean_distance(point, reference_point)  # 计算距离
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point

def calculate_centroid(group):
    if not group:  # 确保组不为空
        return None
    centroid_x = sum(x for x, y in group) / len(group)
    centroid_y = sum(y for x, y in group) / len(group)
    return (centroid_x, centroid_y)

def distance(p, c):
    # p 和 c 是元组形式的点，比如 (x, y)
    x_p, y_p = p
    x_c, y_c = c
    return math.sqrt((x_p - x_c) ** 2 + (y_p - y_c) ** 2)


def solve_tsp_for_each_group(points):
    """针对每个闭合点集选取一个点并应用TSP算法"""
    result = []
    selected_points = []

    for group in points:
        if group:  # 确保该组不为空
            selected_point = closest_point_first(group, selected_points)
            selected_points.append(selected_point)  # 添加选择的点
    # 对选中的点集应用TSP算法
    if(len(points) < 10):
        best_path = brute_force_tsp(selected_points)
    else:
        best_path = christofides_algorithm(selected_points)
    # 关联返回路径
    result.append([selected_points[i] for i in best_path])

    return result
def parse_input():
    """解析输入数据"""
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

    points = parse_input()
    result = solve_tsp_for_each_group(points)
    for group in result:
        print(group)

if __name__ == '__main__':
    main()

# =========
# 8.17分
# 8.92分
# 31.74分
# 16.72分
# 36.27分
# 57.41分
# 49.13分
# 139.80分
# 127.00分
# 261.33分
####第一次得分