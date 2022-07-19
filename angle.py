import math


# 1. 找到4,3,6三个点
# 按顺序存进point_sort列表
def find_points(point_list):
    point_sort = []

    point_list.sort()

    point_sort.append(point_list[3])
    point_sort.append(point_list[0])
    point_sort.append(point_list[5])
    return point_sort


# 2.  计算三边长
def euclidean_distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


# 3. 计算角436的值
def angle(a, b, c):
    """
    a, b: the length of two adjacent edges
    c: the length of opposite site
    """
    d = math.degrees(math.acos((a * a + b * b - c * c) / (2 * a * b)))
    return d


"""
     4
     *
    *  * 
   *    *  
  *      * 
 *        *
************ 
3           6
"""
# INPUT
all_points = [[(1315.011474609375, 406.929443359375),
               (1212.295654296875, 626.8081665039062),
               (1084.995361328125, 752.9243774414062),
               (988.5630493164062, 753.6880493164062),
               (1302.6756591796875, 828.309326171875),
               (738.0346069335938, 831.6212158203125)],

              [(1302.8018798828125, 401.1720275878906),
               (1207.6300048828125, 626.0771484375),
               (1083.2247314453125, 756.1541137695312),
               (986.1935424804688, 757.182861328125),
               (1302.7919921875, 827.3223876953125),
               (736.1265869140625, 837.8878784179688)]]
def return_angle(points_coordinate):
    
    
    point_4 = points_coordinate[1]
    point_3 = points_coordinate[0]
    point_6 = points_coordinate[2]
    
    # 计算对应边长
    edge_34 = euclidean_distance(point_3, point_4)
    edge_36 = euclidean_distance(point_3, point_6)
    edge_46 = euclidean_distance(point_4, point_6)
    return angle(edge_34, edge_36, edge_46)

