import random
import math

N = 10000  # 坐标范围
M = 8 # 图形数量

# 生成随机整数点
def random_point(x_range, y_range):
    return (random.randint(*x_range), random.randint(*y_range))

# 生成矩形的顶点
def generate_rectangle(x_range, y_range):
    x1, y1 = random_point(x_range, y_range)
    width = random.randint(1, 10)
    height = random.randint(1, 10)
    x2, y2 = x1 + width, y1
    x3, y3 = x1 + width, y1 + height
    x4, y4 = x1, y1 + height
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

# 生成三角形的顶点
def generate_triangle(x_range, y_range):
    return [random_point(x_range, y_range) for _ in range(3)]

# 生成圆的整数顶点（取圆周上的整数点）
def generate_circle(x_range, y_range):
    center = random_point(x_range, y_range)
    radius = random.randint(3, 10)
    points = [
        (center[0], center[1] + radius),
        (center[0], center[1] - radius),
        (center[0] + radius, center[1]),
        (center[0] - radius, center[1])
    ]
    return points

# 生成指定数量的图形
def generate_shapes(num_shapes):
    shapes = []
    x_range = (0, N)
    y_range = (0, N)

    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'triangle', 'circle'])

        if shape_type == 'rectangle':
            shapes.append(generate_rectangle(x_range, y_range))
        elif shape_type == 'triangle':
            shapes.append(generate_triangle(x_range, y_range))
        elif shape_type == 'circle':
            shapes.append(generate_circle(x_range, y_range))

    return shapes

# 格式化图形数据为字符串
def format_shapes(shapes):
    return "@".join([str(shape) for shape in shapes])

# 生成并格式化图形
shapes = generate_shapes(M)
formatted_shapes = format_shapes(shapes)
print(formatted_shapes)