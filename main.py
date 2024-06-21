import os
import json
from typing import List
import numpy as np
import random
from collections import deque
from queue import PriorityQueue 
from tqdm import tqdm
import pickle
from copy import deepcopy
# 定义常量
MAP_SIZE = 15               # 地图大小（公里）
J = 3             # 配送中心数量
K = 12          # 卸货点数
T = 20          # 时间间隔（分钟）
T_MAX = 24 * 60        # 总时间（分钟）
M = 3                       # 卸货点生成最多订单数量
N = 8                      # 最多携带物品数量
UAV_MAX_DISTANCE = 20           # 无人机一次飞行最大距离
UAV_SPEED = 1             # 无人机速度（公里/分钟）

POP_SIZE = 50               # 种群大小
G_max = 1000                # 迭代次数
MUTATION_RATE = 0.1         # 变异概率
DELTA = 0.5                 # 适应度多样性阈值 
            
class loading_center(): # 装货点
    def __init__(self, id, location = (0, 0)): # 初始化装货点
        self.location = location
        self.id = id

    def __str__(self):
        return 'L'+str(self.id)        

class unloading_center():
    def __init__(self, id, location = (0, 0)):
        self.location = location
        self.order_list = []
        self.id = id
        self.nearest_loading_center_id = None
    
    def __str__(self):
        return 'U'+str(self.id)

    def generate_order(self):
        # 卸货点会随机生成订单
        priority = random.randint(1, 3)
        if priority == 1:
            time = 180
        elif priority == 2:
            time = 90
        elif priority == 3:
            time = 30
        return time
    
    def step_forward(self, m, t): # 步进结点状态
        # 订单时间更新
        for i in range(len(self.order_list)):
            self.order_list[i] -= t
        # 卸货点会生成n个订单(0-m个订单)
        n = random.randint(0, m)
        for i in range(n): # 生成订单
            self.order_list.append(self.generate_order())
        self.order_list.sort() # 自动排序
    
    def get_imergency_order(self): # 获取最紧急的订单
        return [order for order in self.order_list if order <= T]
    
    def get_order_list(self):
        return self.order_list

    def satify_orders(self, satified_orders):
        for satified_order in satified_orders:
            self.order_list.remove(satified_order)

class MAP():
    def __init__(self):
        self.loading_centers = [] # 装载点坐标
        self.unloading_centers = [] # 卸货点坐标
        self.loading_nodes = [] # 装载点
        self.unloading_nodes = [] # 卸货点
        self.distance_matrix = {'from_loading_center': [], 'from_unloading_center': []}

    def load_map(self, map_info_path = './map_info.pkl', num_loading = 3, num_unloading = 12):
        if os.path.exists(map_info_path):
            # Load map information from file
            with open(map_info_path, 'rb') as file:
                map_info = pickle.load(file)
            
            # 获取各个配送中心和卸货点的坐标
            self.loading_centers = map_info['loading_centers']
            self.unloading_centers = map_info['unloading_centers']

        else:
            print('Map information file does not exist.')
            print('Generating map...')
            self.generate_map(num_loading, num_unloading)
            self.save_map(map_info_path)
            self.show_map(True)

        # 初始化配送中心和卸货点
        self.init_loading_center()
        self.init_unloading_center()

        # 计算配送中心和卸货点之间的距离，初始化距离矩阵
        self.get_distance_matrix()


    def generate_map(self, num_loading = 3, num_unloading = 12):
        loading_centers = []
        unloading_centers = []
        for i in range(num_loading):
            loading_centers.append((random.uniform(0, 10), random.uniform(0, 10)))
        
        while len(unloading_centers) < num_unloading:
            x, y = random.uniform(0, 20), random.uniform(0, 20)
            for center in loading_centers: # 保证卸货点至少在一个装货点附近
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 < 10:
                    unloading_centers.append((x, y))
                    break
                else:
                    continue
        
        self.loading_centers = loading_centers
        self.unloading_centers = unloading_centers

    def save_map(self, map_info_path = './map_info.pkl'):
        # Save map information to file
        map_info = {
            'loading_centers': self.loading_centers,
            'unloading_centers': self.unloading_centers
        }

        with open(map_info_path, 'wb') as file:
            pickle.dump(map_info, file)

    def show_map(self, if_savemap = False):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.scatter(*zip(*self.loading_centers), color='blue', label='Loading Centers')
        plt.scatter(*zip(*self.unloading_centers), color='red', label='Unloading Centers')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('UAV Delivery Map')
        plt.legend()
        plt.grid(True)
        if if_savemap:
            plt.savefig('./map_image.png')
        plt.show()
    
    def get_distance_matrix(self):
        # 计算配送中心到各个卸货点的距离
        for i in range(len(self.loading_centers)):
            self.distance_matrix['from_loading_center'].append([])
            for j in range(len(self.unloading_centers)):
                self.distance_matrix['from_loading_center'][i].append(((self.loading_centers[i][0] - self.unloading_centers[j][0]) ** 2 + (self.loading_centers[i][1] - self.unloading_centers[j][1]) ** 2) ** 0.5)
        # 计算各个卸货点之间的距离
        for i in range(len(self.unloading_centers)):
            self.distance_matrix['from_unloading_center'].append([])
            for j in range(len(self.unloading_centers)):
                self.distance_matrix['from_unloading_center'][i].append(((self.unloading_centers[i][0] - self.unloading_centers[j][0]) ** 2 + (self.unloading_centers[i][1] - self.unloading_centers[j][1]) ** 2) ** 0.5)
        for i in range(len(self.unloading_nodes)):
            self.unloading_nodes[i].nearest_loading_center_id = [self.distance_matrix['from_loading_center'][j][i] for j in range(0,3)].index(min([self.distance_matrix['from_loading_center'][j][i] for j in range(0,3)]))

    def init_loading_center(self):
        for i in range(len(self.loading_centers)):
            self.loading_nodes.append(loading_center(i, self.loading_centers[i]))
    
    def init_unloading_center(self):
        for i in range(len(self.unloading_centers)):
            self.unloading_nodes.append(unloading_center(i, self.unloading_centers[i]))

    def show_path(self, routings):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        for routing in range(len(routings)):
            node_coordinates = []
            for i in range(len(routings[routing])):
                node_coordinates.append(routing[i].location)
            # 根据node中的结点坐标绘制路线

        # 提取x坐标和y坐标
        x = [coord[0] for coord in node_coordinates]
        y = [coord[1] for coord in node_coordinates]

        # 绘制路线
        plt.plot(x, y, marker='o')

        # 添加节点标签
        for i, coord in enumerate(node_coordinates):
            plt.text(coord[0], coord[1], f'Node {i+1}', ha='center', va='bottom')

        # 设置图形标题和坐标轴标签
        plt.title('Node Route')
        plt.xlabel('X')
        plt.ylabel('Y')

        # 显示图形
        plt.show()

map = MAP()
map.load_map("./map_info.pkl")

# 路径
class Routes:
    def __init__(self, routes = []):
        self.fitness = 0
        self.total_distance = 0
        self.sub_routes = routes # 该方案下所有的子路径
    
    def show_routes(self, map):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.scatter(*zip(*map.loading_centers), color='blue', label='Loading Centers')
        plt.scatter(*zip(*map.unloading_centers), color='red', label='Unloading Centers')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('UAV Delivery Map')
        plt.legend()
        plt.grid(True)
        for route in self.sub_routes:
            s_e = map.loading_nodes[map.unloading_nodes[route[0]].nearest_loading_center_id]
            nodes = []

            for k in route:
                nodes.append(map.unloading_nodes[k])

            plt.plot([s_e.location[0],nodes[0].location[0]],[s_e.location[1],nodes[0].location[1]])
            for i in range(len(nodes)-1):
                plt.plot([nodes[i].location[0],nodes[i+1].location[0]],[nodes[i].location[1],nodes[i+1].location[1]])
            
            plt.plot([nodes[-1].location[0],s_e.location[0]],[nodes[-1].location[1],s_e.location[1]])
        plt.show()

    def print_routes(self, unloading_points):
        for route in self.sub_routes:
            center_id = unloading_points[route[0]].nearest_loading_center_id
            print("路径："+ centers[center_id].__str__() + "->" + "".join([unloading_points[k].__str__() + "->" for k in route]) + centers[center_id].__str__())
                


class Search():
    def __init__(self, map):
        self.map = map

    def get_emergency_order_list(self):
        emergency_nodes = [] # 紧急订单的卸货点
        emergency_orders = [0]*K # 紧急订单
        emergency_orders_detail = [[]]*K
        for unloading_point in self.map.unloading_nodes:
            emergency_order = unloading_point.get_imergency_order()
            if emergency_order:
                print(f"卸货点{unloading_point.id}有紧急订单：{emergency_order}")
                emergency_nodes.append(unloading_point.id)
                emergency_orders_detail[unloading_point.id] = emergency_order
                emergency_orders[unloading_point.id]=len(emergency_order)
        if emergency_nodes:
            return emergency_nodes, emergency_orders, emergency_orders_detail
        else:
            return None

    def routes_to_permutation(self, routes):
        permutation = []
        for route in routes.sub_routes:
            permutation += route
        return permutation
        
    def calculate_fitness(self, routes, distance_matrix, payloading = 0):
        total_distance = 0
        total_items = 0
        for route in routes.sub_routes:
            center_id = unloading_points[route[0]].nearest_loading_center_id

            total_distance += map.distance_matrix['from_loading_center'][center_id][route[0]]

            for i in range(len(route)-1):
                total_items += len(unloading_points[i].order_list)
                total_distance += distance_matrix[route[i]][route[i+1]]
                
            total_items += len(unloading_points[route[-1]].order_list)
            total_distance += map.distance_matrix['from_loading_center'][center_id][route[-1]]
        routes.total_distance = total_distance
        fitness = total_distance / (payloading + 0.01) # 适应度的计算最终是平均运送每一个货物需要的距离，越低越好
        fitness = fitness
        return fitness
    
    def initialize_population(self, distance_matrix, current_time):
        population = []

        # 紧急订单
        emergency_list = self.get_emergency_order_list()

        # 精英解生成
        elite_solution = self.savings_algorithm(distance_matrix, centers, unloading_points, current_time, emergency_list)
        elite_permutation = search.routes_to_permutation(elite_solution)

        # 添加精英解到种群
        fitness = self.calculate_fitness(elite_solution, distance_matrix, payloading= 0)
        population.append((elite_permutation, fitness))

        # 随机生成剩余解
        delta = DELTA
        count = 0
        while len(population) < POP_SIZE:
            individual = elite_permutation[:]
            random.shuffle(individual)
            routes = self.permutation_to_routes(individual, distance_matrix)
            fitness = self.calculate_fitness(routes, distance_matrix)
            
            # 检查多样性
            if all(abs(fitness - ind[1]) >= delta for ind in population):
                population.append((individual, fitness))
            if count >= G_max: 
                count = 0
                delta -= 0.1
            count += 1
        return population

    def savings_algorithm(self, distance_matrix, centers, unloading_points, current_time, emergency_list):
        must_dispatch_points = emergency_list[0]
        print("当前时间必须去的卸货点：", must_dispatch_points)
        routes = [[node] for node in emergency_list[0]]
        savings = []

        for i in range(K):
            for j in range(i + 1, K):
                id_i = unloading_points[i].id
                id_j = unloading_points[j].id
                nearest_center_id = unloading_points[i].nearest_loading_center_id
                saving = (map.distance_matrix['from_loading_center'][nearest_center_id][id_i] + map.distance_matrix['from_loading_center'][nearest_center_id][id_j] - distance_matrix[id_i][id_j])
                savings.append((saving, id_i, id_j, nearest_center_id))
                
        savings.sort(reverse=True)

        # 节约算法
        # 约束条件1：路程上任何一个点都不能有货物超过配送时间
        # 约束条件2：飞机容量约束
        # 约束条件3：距离约束
        while(len(savings) > 0):
        # for saving, id_i, id_j in savings:
            saving, id_i, id_j, nearest_center_id = savings[0]
            savings = savings[1:]
            route_i = None
            route_j = None

            for route in routes:
                if id_i == route[-1]:   # 确保配送中心不变
                    route_i = route
                if id_j == route[0]:
                    route_j = route
            
            if route_i == None or route_j == None: continue

            if route_i is not None and route_j is not None and route_i != route_j:
                new_route = route_i + route_j
                total_items = sum(emergency_list[1][point_id] for point_id in new_route)
                if total_items > N: continue
                
                center_id = unloading_points[new_route[0]].nearest_loading_center_id
                distance_list = [map.distance_matrix['from_loading_center'][center_id][new_route[0]]]
                for k in range(len(new_route) - 1):
                    distance_list.append(distance_matrix[new_route[k]][new_route[k + 1]])
                distance_list.append(map.distance_matrix['from_loading_center'][center_id][new_route[-1]])
                total_distance = sum(distance_list)
                if total_distance > UAV_MAX_DISTANCE: continue
                
                dispatch_time = [(sum(distance_list[:k+1]) / UAV_SPEED) for k in range(len(new_route))]
                deadline_list = [emergency_list[2][node_id][0] for node_id in new_route]
                if sum([dispatch_time[i] > deadline_list[i] for i in range(len(deadline_list))]): continue

                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(new_route)
                
                # 更新节约值，因为合并路径后有些路径的配送中心发生改变
                for s, i, j, c in savings:
                    if i in new_route and j in new_route:
                        savings.remove((s, i, j, c))
                    elif j in new_route[1:]:
                        savings.remove((s, i, j, c))
                    elif i in new_route[:-1]:
                        savings.remove((s, i, j, c))
                    elif i == new_route[-1]:
                        savings.remove((s, i, j, c))
                        saving = (map.distance_matrix['from_loading_center'][nearest_center_id][i] + map.distance_matrix['from_loading_center'][nearest_center_id][j] - map.distance_matrix['from_unloading_center'][i][j])
                        savings.append((saving, i, j, nearest_center_id))
                savings.sort(reverse=True)

        return Routes(routes)

    def mutate(self, child, child_routes):  
        better_cost =child_routes.total_distance
        tmp_cost = 0

        random_up_index = random.randint(0, len(child)-1)
        random_up = child[random_up_index]
        
        # 找到随机选择的卸货点的路段
        random_up_route = []
        for route in child_routes.sub_routes:
            if random_up in route:
                random_up_route = route
                break
            
        v1 = v2 = s1 = s2 = 0
        if random_up_index > 0:
            v1 = child[random_up_index - 1]
        if random_up_index < len(child) - 1:
            v2 = child[random_up_index + 1]
        
        if v1 and v2 and random_up != random_up_route[0]:
            s1 = map.distance_matrix['from_unloading_center'][random_up][v1] + map.distance_matrix['from_unloading_center'][random_up][v2] - map.distance_matrix['from_unloading_center'][v1][v2]
        else:
            s1 = map.distance_matrix['from_loading_center'][unloading_points[random_up].nearest_loading_center_id][random_up]  + map.distance_matrix['from_unloading_center'][random_up][v2] - map.distance_matrix['from_loading_center'][unloading_points[v2].nearest_loading_center_id][v2]

        # 尝试在其他路径中插入这个卸货点
        route_index = better_route_index = 0
        for route in child_routes.sub_routes:
            if route != random_up_route and sum([len(unloading_points[point].order_list) for point in route]) + len(unloading_points[random_up].order_list) <= N:
                for i in range(len(route)):
                    v1 = route[0] if i == 0 else route[i - 1]
                    v2 = route[i]
                    new_route = route[:i] + [random_up] + route[i:]
                    
                    if i == 0:
                        center_id = unloading_points[random_up].nearest_loading_center_id
                        distance_list = [map.distance_matrix['from_loading_center'][center_id][random_up]]
                    else:
                        center_id = unloading_points[route[0]].nearest_loading_center_id 
                        distance_list = [map.distance_matrix['from_loading_center'][center_id][route[0]]]
                    for k in range(len(new_route) - 1):
                        distance_list.append(map.distance_matrix['from_unloading_center'][new_route[k]][new_route[k + 1]])
                    distance_list.append(map.distance_matrix['from_loading_center'][center_id][new_route[-1]])
                    
                    total_distance_tmp = sum(distance_list)
                    if total_distance_tmp > UAV_MAX_DISTANCE: continue
                    
                    dispatch_time = [current_time + (sum(distance_list[:k+1]) / UAV_SPEED) for k in range(len(new_route)-1)]
                    deadline_list = [unloading_points[k].order_list[0] for k in new_route]
                    if dispatch_time > deadline_list: continue
                
                    if i == 0:
                        s2 = unloading_points[random_up].nearest_center_distance + map.distance_matrix['from_unloading_center'][random_up][v2] - unloading_points[route[0]].nearest_center_distance
                    else:
                        s2 = map.distance_matrix['from_unloading_center'][v1][random_up] + map.distance_matrix['from_unloading_center'][v2][random_up] - map.distance_matrix['from_unloading_center'][v1][v2]
                    tmp_cost = (better_cost - s1 - s2)

                    if tmp_cost < better_cost:
                        better_cost = tmp_cost
                        next_up_index = i
                        better_route_index = route_index
            route_index += 1
            
        # 如果找到更好的路径，进行插入操作
        if better_cost < child_routes.total_distance:
            random_up_route.remove(random_up)
            child_routes.sub_routes[better_route_index].insert(next_up_index, random_up)
            if [] in child_routes.sub_routes:
                child_routes.sub_routes.remove([]) # 删除空路径
            fitness = calculate_fitness(child_routes, map.distance_matrix['from_unloading_center'])
            
    def generate_child(self, child, parent1, parent2):
        N = len(parent1)
        c1, c2 = [0] * N, [0] * N
        flag1, flag2 = [0] * (K), [0] * (K)
        
        i = random.randint(0, N - 1)
        j = i
        while j == i:
            j = random.randint(0, N - 1)
        if j < i:
            i, j = j, i
        
        for k in range(i, j + 1):
            c1[k] = parent1[k]
            flag1[c1[k]] = True
            c2[k] = parent2[k]
            flag2[c2[k]] = True
        
        index = (j + 1) % N
        p1, p2 = index, index
        while index != i:
            while flag1[parent2[p2]]:
                p2 = (p2 + 1) % N
            while flag2[parent1[p1]]:
                p1 = (p1 + 1) % N
            c1[index] = parent2[p2]
            flag1[c1[index]] = True
            c2[index] = parent1[p1]
            flag2[c2[index]] = True
            index = (index + 1) % N

        if random.random() < 0.5:
            child[:] = c1
        else:
            child[:] = c2

    def permutation_to_routes(self, permutation, distance_matrix):
        routes = []
        current_route = None
        for point_id in permutation:
            unloading_point = unloading_points[point_id]
            if not current_route:
                current_route = [point_id]
                center_id = unloading_point.nearest_loading_center_id
                total_items = len(unloading_point.order_list)
                distance_list = [map.distance_matrix['from_loading_center'][center_id][point_id]]
                dispatch_time = current_time + (distance_list[0] / UAV_SPEED)
            else:
                total_items += len(unloading_point.order_list)
                if total_items > N: 
                    routes.append(current_route)
                    current_route = [point_id]
                    total_items = dispatch_time = 0
                    distance_list = []
                    continue
                distance_list.append(distance_matrix[current_route[-1]][point_id])
                total_distance = sum(distance_list)
                if total_distance > UAV_MAX_DISTANCE: 
                    routes.append(current_route)
                    current_route = [point_id]
                    continue
                dispatch_time += (distance_matrix[current_route[-1]][point_id] / UAV_SPEED)
                if dispatch_time > unloading_point.order_list[0]:
                    routes.append(current_route)
                    current_route = [point_id]
                    continue
                current_route.append(point_id)
        routes.append(current_route)
        res = Routes(routes)
        self.calculate_fitness(res, distance_matrix, 0)
        return res

search = Search(map)
if __name__ == "__main__":
    # 确保输出目录存在
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    with open("./map_info.pkl", 'rb') as file:
        map_info = pickle.load(file)
    file.close()
    loading_centers = map.loading_centers
    unloading_centers = map.unloading_centers

    centers = map.loading_nodes
    unloading_points = map.unloading_nodes

    current_orders = deque()
    current_time = 0
    total_distance = 0
    time_list = []
    total_distance_list = []
    while current_time < T_MAX:
        time_list.append(current_time)
        print("当前时间：",current_time)
        for point in unloading_points:
            point.step_forward(M, T) # 自动更新目前订单情况
            point.generate_order() # 自动生成新订单
            print(f"卸货点{point.id}订单情况为：{point.get_order_list()}") # 打印订单情况

        # 先判断有没有必须去的卸货点
        must_dispatch_points = search.get_emergency_order_list()
        if must_dispatch_points == None:
            print("当前时间没有必须去的卸货点")
            current_time += T
            total_distance_list.append(total_distance)
            continue

        # 初始化种群
        population = search.initialize_population(map.distance_matrix['from_unloading_center'], current_time)
        
        # 先处理超过无人机容量限制的卸货点
        # 但是当前常量定义下不会有超过无人机容量的卸货点，就不处理了

        # 初始化种群
        best_solution = population[0][0]
        best_fitness = float('inf')
            
        delta = DELTA

        # 主循环
        # for generation in tqdm(range(G_max),total=G_max,desc = "迭代进度：",ncols = 100,postfix = dict,mininterval = 0.3):
        
        if len(best_solution) > 1:
            for generation in range(G_max):
                # 随机选择父母，产生子代
                count = 0
                while(1):
                    parent1, parent2 = random.sample(population, 2)
                    parent1 = parent1[0]
                    parent2 = parent2[0]
                    child = [0] * len(parent1) 
                    search.generate_child(child, parent1, parent2)
                    child_routes = search.permutation_to_routes(child, map.distance_matrix['from_unloading_center'])
                    fitness = search.calculate_fitness(child_routes, map.distance_matrix['from_unloading_center'])
                    
                    # 变异：局部搜索改进
                    if random.random() < MUTATION_RATE:           
                        search.mutate(child, child_routes)
                    # 检查多样性
                    if all(abs(fitness - ind[1]) >= delta for ind in population):
                        population.append((child, fitness))
                        break
                    else:
                        count += 1
                        if count >= G_max: 
                            count = 0
                            delta -= 0.1
                    
                # 选出被淘汰的个体
                population.sort(key=lambda ind: ind[1])
                population = population[:POP_SIZE]
                
            best_solution = population[0][0]
        best_routes = search.permutation_to_routes(best_solution, map.distance_matrix['from_unloading_center'])  
        
        # 删去不必要的路径
        final_routes = []
        for route in best_routes.sub_routes:
            if route and any(point in must_dispatch_points[0] for point in route): # unloading_points[route[0]].nearest_center_id 
                nearest_center_id = unloading_points[route[0]].nearest_loading_center_id
                final_routes.append(route)
              
        best_routes = Routes(final_routes)
        # best_routes.show_routes(map)
        search.calculate_fitness(best_routes, map.distance_matrix['from_unloading_center'])
        best_routes.print_routes(unloading_points)   
        total_distance += best_routes.total_distance 
        total_distance_list.append(total_distance)
        

        # 清空队列
        emergency_order_list = search.get_emergency_order_list()
        for up in best_solution:
            unloading_points[up].satify_orders(emergency_order_list[2][up])
          
        current_time += T
    
    print("总路程：", total_distance)