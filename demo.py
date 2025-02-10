import open3d as o3d
import numpy as np


def read_ply_file(file_path):
    """读取PLY文件并返回Open3D点云对象"""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def write_ply_file(pcd, file_path):
    """将Open3D点云对象写入PLY文件"""
    o3d.io.write_point_cloud(file_path, pcd)


def remove_intersecting_points(pcd_a, pcd_b, threshold=0.01):
    tree_b = o3d.geometry.KDTreeFlann(pcd_b)
    remove_indices = []

    # 将pcd_a的点转换为NumPy数组以便迭代
    points_a = np.asarray(pcd_a.points)

    for i in range(len(points_a)):
        # 对pcd_a中的每个点执行最近邻搜索
        [_, idx, dist] = tree_b.search_knn_vector_3d(points_a[i], 1)
        # 如果最近邻的距离小于阈值，则记录该点的索引
        if dist[0] < threshold:
            remove_indices.append(i)

    # 使用Open3D的select_by_index方法移除点
    # 注意：我们需要使用不在remove_indices中的点的索引
    keep_indices = np.array([i for i in range(len(pcd_a.points)) if i not in remove_indices])
    pcd_a_filtered = pcd_a.select_by_index(keep_indices)

    return pcd_a_filtered


def remove_subset_point_cloud(pcd_b, pcd_a, tolerance=1e-6):
    """
    从pcd_b中删除按顺序出现的pcd_a点云。

    参数:
    pcd_b (numpy.ndarray): 原始点云，形状为(N, 3)，其中N是点的数量。
    pcd_a (numpy.ndarray): 要删除的子点云，形状为(M, 3)，其中M是点的数量。
    tolerance (float): 点匹配时的容差。

    返回:
    numpy.ndarray: 删除pcd_a后的pcd_b点云。
    """
    # 确保pcd_a和pcd_b是NumPy数组
    pcd_b = np.asarray(pcd_b.points)
    pcd_a = np.asarray(pcd_a.points)

    # 初始化起始和结束索引
    start_idx = -1
    end_idx = -1

    # 遍历pcd_b，查找pcd_a的起始位置
    for i in range(len(pcd_b) - len(pcd_a) + 1):
        if np.allclose(pcd_b[i:i + len(pcd_a)], pcd_a, atol=tolerance):
            start_idx = i
            break

    # 如果找到了起始位置，则设置结束索引并删除对应部分
    if start_idx != -1:
        end_idx = start_idx + len(pcd_a)
        pcd_b_result = np.concatenate((pcd_b[:start_idx], pcd_b[end_idx:]))
        return pcd_b_result
    else:
        # 如果没有找到匹配的pcd_a，则返回原始的pcd_b
        return pcd_b

# 读取点云文件
pcd_a = read_ply_file('file2.ply')
pcd_b = read_ply_file('file1.ply')

# 执行点云相减操作
pcd_result = remove_subset_point_cloud(pcd_a, pcd_b, tolerance=0.01)

# 保存结果点云到文件
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_result)
write_ply_file(pcd, 'path_to_pcd_result.ply')
