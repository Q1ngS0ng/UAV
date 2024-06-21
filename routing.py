'''
Author: YuwanZ
Date: 2024-06-04 15:51:05
LastEditors: YuwanZ
LastEditTime: 2024-06-18 17:08:27
Description:        
            解决无人机配送路径规划问题
            无人机可以快速解决最后10公里的配送,本作业要求设计一个算法,实现如下图所示区域的无人机配送的路径规划。
            在此区域中,共有j个配送中心,任意一个配送中心有用户所需要的商品,其数量无限,同时任一配送中心的无人机数量无限。
            该区域同时有k个卸货点(无人机只需要将货物放到相应的卸货点即可),假设每个卸货点会随机生成订单,
            一个订单只有一个商品,但这些订单有优先级别,分为三个优先级别(用户下订单时,会选择优先级别,优先级别高的付费高):
                	一般:3小时内配送到即可;
                	较紧急:1.5小时内配送到;
                	紧急:0.5小时内配送到。

            我们将时间离散化,也就是每隔t分钟,所有的卸货点会生成订单(0-m个订单),同时每隔t分钟,系统要做成决策,包括:
            1. 哪些配送中心出动多少无人机完成哪些订单;
            2. 每个无人机的路径规划,即先完成那个订单,再完成哪个订单,...,最后返回原来的配送中心;
            注意:系统做决策时,可以不对当前的某些订单进行配送,因为当前某些订单可能紧急程度不高,可以累积后和后面的订单一起配送。

            目标:一段时间内(如一天),所有无人机的总配送路径最短
            约束条件:满足订单的优先级别要求

            假设条件:
            1. 无人机一次最多只能携带n个物品;
            2. 无人机一次飞行最远路程为20公里(无人机送完货后需要返回配送点);
            3. 无人机的速度为60公里/小时;
            4. 配送中心的无人机数量无限;
            5. 任意一个配送中心都能满足用户的订货需求;
FilePath: /UAV/routing.py
'''
# -*- coding: utf-8 -*-

import random
import pickle
import os

# 全局变量
j = 3   # 配送中心个数
k = 12 # 卸货点个数
m = 1 # 每个时间间隔生成订单最大个数
t = 20 # 时间间隔
t_max = 1440 # 一天时间

class loading_center(): # 装货点
    def __init__(self, location = (0, 0)): # 初始化装货点
        self.location = location

class unloading_center():
    def __init__(self, location = (0, 0)):
        self.location = location
        self.order_list = []
    
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
    
    def get_order_list(self):
        return self.order_list

class MAP():
    def __init__(self):
        self.loading_centers = []
        self.unloading_centers = []
        self.loading_nodes = []
        self.unloading_nodes = []
        self.distance_matrix = {'from_loading_center': {}, 'from_unloading_center': {}}

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

        # 计算配送中心和卸货点之间的距离，初始化距离矩阵
        self.get_distance_matrix()
        # 初始化配送中心和卸货点
        self.init_loading_center()
        self.init_unloading_center()

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
            for j in range(len(self.unloading_centers)):
                self.distance_matrix['from_loading_center'][(i, j)] = ((self.loading_centers[i][0] - self.unloading_centers[j][0]) ** 2 + (self.loading_centers[i][1] - self.unloading_centers[j][1]) ** 2) ** 0.5
        # 计算各个卸货点之间的距离
        for i in range(len(self.unloading_centers)):
            for j in range(len(self.unloading_centers)):
                self.distance_matrix['from_unloading_center'][(i, j)] = ((self.unloading_centers[i][0] - self.unloading_centers[j][0]) ** 2 + (self.unloading_centers[i][1] - self.unloading_centers[j][1]) ** 2) ** 0.5

    def init_loading_center(self):
        for i in range(len(self.loading_centers)):
            self.loading_nodes.append(loading_center(self.loading_centers[i]))
    
    def init_unloading_center(self):
        for i in range(len(self.unloading_centers)):
            self.unloading_nodes.append(unloading_center(self.unloading_centers[i]))

class uavs():
    # 无人机有种种限制，但是实际上无人自从派出去的那一刻起，就相当于任务结束了，只需要记录其路径即可，然后返回一个总距离，作为其消耗。
    # 无限个无人机，不就相当于一个无人机出发了n次吗？
    # 简单来说，就是一个瞬时，一个无人机，从一个配送点出发了n次（题目是理想的，因此可以不记录时间，不考虑无人机数量限制）
    # 那么无人机就应该只处理当前时间节点下，最紧急的订单。
    # 无人机的路径规划，就是一个最短路径问题，可以用遗传算法解决。
    def __init__(self, max_load = 3, max_distance = 20, speed = 1):
        self.max_load = max_load
        self.max_distance = max_distance
        self.speed = speed
        self.location = (0, 0)
        self.order_list = []
        self.route = []
        self.payload = 0
    
    def record_route(self, node):
        self.route.append(node)

    def sum_distence(self, distence):
        pass

class Search():
    pass

if __name__ == '__main__':
    
    # 生成地图，并展示
    map = MAP()
    # map.generate_map(3, 12)
    # map.show_map()
    # map.save_map("./map_info.pkl")
    map.load_map("./map_info.pkl")
    map.show_map()
    
    for i in range(t_max // t):
        # 卸货点更新订单
        for node in map.unloading_nodes:
            node.step_forward(m, t)
        # 系统决策
        
        # 1. 哪些配送中心出动多少无人机完成哪些订单;
        # 2. 每个无人机的路径规划,即先完成那个订单,再完成哪个订单,...,最后返回原来的配送中心;
        # 3. 可以不对当前的某些订单进行配送,因为当前某些订单可能紧急程度不高,可以累积后和后面的订单一起配送。
        
    pass
