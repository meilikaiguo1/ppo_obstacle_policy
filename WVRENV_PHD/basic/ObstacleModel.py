"""
    输入当前飞机经纬度，返回当前飞机周围位置的障碍物经纬高
"""
import math

from scipy.spatial import KDTree
from WVRENV_PHD.utils.GNCData import ned_to_wgs84,wgs84ToNED
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sensor_pargs = [10, 2000, 1, 10]

# 设置采样点数，注意 SRTM1 数据需要 3601 个点
SAMPLES = 1201
f_name = "./N32E096.hgt"

originLongitude = 96
originLatitude = 32

# 读取 HGT 原始高程
with open(f_name, 'rb') as hgt_data:
    elevations_raw = np.fromfile(
        hgt_data,
        np.dtype('>i2'),
        SAMPLES * SAMPLES
    ).reshape((SAMPLES, SAMPLES))

# =========================================================
# 关键修改：
# HGT 文件第一行通常对应北侧，也就是 33°N；
# 但是这里 Latitude_list 使用 32 -> 33 递增，
# 所以必须将 elevations_raw 上下翻转。
# =========================================================
elevations = np.flipud(elevations_raw)

Latitude_list = np.linspace(originLatitude, originLatitude + 1, SAMPLES)
Longitude_list = np.linspace(originLongitude, originLongitude + 1, SAMPLES)

# 地形插值器
elev_interpolator = RegularGridInterpolator(
    (Latitude_list, Longitude_list),
    elevations,
    method='linear',
    bounds_error=False,
    fill_value=None
)


def get_elevation(lat, lon):
    return float(elev_interpolator([[lat, lon]])[0])


def get_elevation_batch(latitudes, longitudes):
    query_points = np.column_stack((latitudes, longitudes))
    return elev_interpolator(query_points)

def ned_to_wgs84_batch(ned_coords):
    """
    批量 NED -> WGS84
    ned_coords: [N, 3]
    return:
        lon_all: [N]
        lat_all: [N]
        alt_all: [N]
    """
    n = ned_coords.shape[0]
    lon_all = np.empty(n, dtype=np.float64)
    lat_all = np.empty(n, dtype=np.float64)
    alt_all = np.empty(n, dtype=np.float64)

    for i in range(n):
        lon, lat, alt = ned_to_wgs84(ned_coords[i])
        lon_all[i] = lon
        lat_all[i] = lat
        alt_all[i] = alt

    return lon_all, lat_all, alt_all
def wgs84_to_ned_batch(wgs84_coords):
    # 假设 wgs84_coords 是一个包含所有扫描点的坐标的数组，形状为 (N, 3)，N 是总扫描点数
    if wgs84_coords.shape[1] != 3:
       return
    point_x, point_y, point_z = wgs84_coords[:, 0], wgs84_coords[:, 1], wgs84_coords[:, 2]
    # 假设 wgs84ToNED 可以批量计算所有点的 NED 坐标
    ned_coords = []

    for x, y, z in zip(point_x, point_y, point_z):
        north, east, down = wgs84ToNED(x, y, z)
        ned_coords.append([north, east, down])

    return np.array(ned_coords)
data_list = []
for i in range(len(Latitude_list)):
    for j in range(len(Longitude_list)):
        data = {
            "latitude": Latitude_list[i],
            "longitude": Longitude_list[j],
            "elevation": elevations[i][j]
        }
        data_list.append(data)

points = np.array([[point["latitude"], point["longitude"]] for point in data_list])
tree = KDTree(points)


def wgs84_to_ned_batch(wgs84_coords):
    """
    wgs84_coords: shape = [N, 3]
    每一行为 [lat, lon, alt]
    返回 NED: [north, east, down]
    """
    wgs84_coords = np.asarray(wgs84_coords, dtype=np.float64)

    if wgs84_coords.ndim != 2 or wgs84_coords.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)

    ned_coords = []

    for lat, lon, alt in wgs84_coords:
        north, east, down = wgs84ToNED(lat, lon, alt)
        ned_coords.append([north, east, down])

    return np.asarray(ned_coords, dtype=np.float64)


class ObstacleEncoder:
    def __init__(self, max_obstacles=10, max_distance=1000):
        self.max_obstacles = max_obstacles  # 每个状态保留的最大障碍点数
        self.max_distance = max_distance    # 考虑障碍物的最大距离（米）

    def encode(self, current_pos, raw_obstacles):

        """
        输入：
        - current_pos: 当前战斗机NED坐标 (n, e, d)
        - raw_obstacles: 原始障碍点列表 [[lon1, lat1, alt1], ...]
        输出：
        - 障碍点的ned坐标
        """
        ned_obstacles=wgs84_to_ned_batch(raw_obstacles)
        # print('ned_obstacles = ',ned_obstacles)
        obstacles=[]
        for i in range(len(ned_obstacles)):
            ned_obstacle=self.relative_ned(ned_obstacles[i],current_pos)
            obstacles.append(ned_obstacle)

        # 过滤距离过远的障碍物
        filtered = [p for p in obstacles
                    if np.linalg.norm(p[:3]) < self.max_distance]

        # 按距离排序并截断
        sorted_obs = sorted(filtered, key=lambda x: np.linalg.norm(x[:3]))[:self.max_obstacles]
        encoded=[]
        for p in sorted_obs:
            encoded.append([
                p[0]+current_pos[0],
                p[1]+current_pos[1],
                p[2]+current_pos[2]
            ])
        return np.array(encoded)
    def encode_rel_ned(self, current_pos, raw_obstacles):

        """
        输入：
        - current_pos: 当前战斗机NED坐标 (n, e, d)
        - raw_obstacles: 原始障碍点列表 [[lon1, lat1, alt1], ...]
        """
        ned_obstacles=wgs84_to_ned_batch(raw_obstacles)
        obstacles=[]
        for i in range(len(ned_obstacles)):
            ned_obstacle=self.relative_ned(ned_obstacles[i],current_pos)
            obstacles.append(ned_obstacle)

        # 过滤距离过远的障碍物
        filtered = [p for p in obstacles
                    if np.linalg.norm(p[:3]) < self.max_distance]

        # 按距离排序并截断
        sorted_obs = sorted(filtered, key=lambda x: np.linalg.norm(x[:3]))[:self.max_obstacles]
        encoded=[]
        for p in sorted_obs:
            encoded.extend([
                np.arctan2((p[1]),(p[0])),
                np.arctan2((p[2]),(p[0])),
                np.linalg.norm(p)
            ])

        return np.array(encoded)

    def relative_ned(self, obstacle_point, current_ned):
        """将障碍点坐标转换为相对当前NED坐标"""
        delta_east = obstacle_point[0] - current_ned[0]
        delta_north = obstacle_point[1] - current_ned[1]
        delta_alt = obstacle_point[2] - current_ned[2]
        return [delta_east, delta_north, delta_alt]

def unit(v,eps:float=1e-9):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("ray_terrain_intersection: 速度向量过小，无法确定射线方向。")
    return v / n



def ray_terrain_intersection(fighter, step_m, max_range, accuracy, max_iter):
    '''
    用二分法找出飞行器速度方向上的障碍点
    :param fighter: 飞行器智能体
    :param step_m: 每次扫描弧长
    :param max_range: 扫描最大范围
    :param accuracy: 精确度
    :param max_iter: 最大迭代次数
    :return:
    '''

    fighter_ned = fighter.state.ned_Pos
    vel_ned = [fighter.fc_data.fNorthVelocity, fighter.fc_data.fEastVelocity, fighter.fc_data.fVerticalVelocity]
    vel_norm = unit(vel_ned)
    Done = False


    def sample(s):
        '''
        :param s:弧长
        :return: 高度差， 经纬高， 地形高
        '''
        p_ned = fighter_ned + s * vel_norm
        p_lon, p_lat, p_alt = ned_to_wgs84(p_ned)
        elev = get_elevation(p_lat, p_lon)
        h = p_alt - elev
        return h, p_ned, p_lon, p_lat, p_alt, elev

    # step1: 起点开始找到变号区间
    h0, p_ned0, lon0, lat0, alt0, elev0 = sample(0)

    # 检测当前位置是否碰撞
    if h0 <= 5:
        Done = True

    s = step_m
    hit_interval = None
    while s <= max_range:
        h, p_ned, p_lon, p_lat, p_alt, elev = sample(s)
        if h <= 0:
            hit_interval = [s - step_m, s]
            break
        s += step_m

    #step2:找到变号区间使用二分法细化
    if hit_interval !=  None:
        lon_mid = lat_mid = elev_mid = None
        s_start = hit_interval[0]
        s_end = hit_interval[1]

        h_s, p_s, lon_s, lat_s, alt_s, elev_s = sample(s_start)
        h_e, p_e, lon_e, lat_e, alt_e, elev_e = sample(s_end)
        for _ in range (max_iter):
            mid = (s_start + s_end) / 2
            h_mid, p_mid, lon_mid, lat_mid, alt_mid, elev_mid = sample(mid)

            if s_end - s_start <= accuracy or abs(h_mid) < 2:
                break
            if h_mid * h_s < 0:
                s_end, h_e = mid, h_mid
            else:
                s_start, h_s = mid, h_mid

        return lon_mid,lat_mid,elev_mid, Done
    else:
        return None,None,None, Done






def NeedtoAvoidObstacle(fighter):
    D2R = math.pi / 180.0
    factor_safety = 1.3
    k_rate_distance = 0.0

    turning_radius = ((fighter.fc_data.fMachNumber - 0.4) * (900.0 - 800.0) / (0.8 - 0.4) + 1200.0)

    current_ned = fighter.state.ned_Pos


    v_range = 0.0
    dis_obstacle = 3000.0
    f_range = 1000 * 1.5
    n_range = 100
    step_range = f_range / n_range
    alt_f = fighter.fc_data.fAltitude
    v_lon, v_lat, v_ele, _ = ray_terrain_intersection(fighter, sensor_pargs[0], sensor_pargs[1], sensor_pargs[2], sensor_pargs[3])
    if v_lon == None:
        return 0.0
    else:
        v_n, v_e, v_d = wgs84ToNED(v_lat, v_lon, v_ele)
        los_vec = [v_n - current_ned[0], v_e - current_ned[1], v_d - current_ned[2]]
        distance = np.linalg.norm(los_vec)

        delta_ele2 = distance * math.sin(np.deg2rad(fighter.fc_data.fPathPitchAngle)) - (alt_f - v_ele)
        x_tmp = delta_ele2 * math.cos(np.deg2rad(fighter.fc_data.fPathPitchAngle))
        inside_sqrt = turning_radius ** 2  - (turning_radius - x_tmp)**2
        inside_sqrt = max(inside_sqrt, 0.0)
        tan_pp = math.tan(np.deg2rad(fighter.fc_data.fPathPitchAngle))
        if abs(tan_pp) < 1e-8:
            d_tmp = float("inf")
        else:
            d_tmp = math.sqrt(inside_sqrt) + x_tmp / tan_pp
        if distance > d_tmp * factor_safety:
            return 0.0
        else:
            return 1.0




    # if v_elevation == None:
    #     v_elevation = 0
    # p_ned = current_ned + v_range * vel_norm
    # p_lon, p_lat, p_alt = ned_to_wgs84(p_ned)
    # # v_elevation = float(get_elevation(p_lat, p_lon))
    # for _ in range(1, n_range):
    #     v_range += step_range
    #     delta_ele2 = v_elevation - alt_f - v_range * math.sin(vel_norm)
    #     x_tmp = delta_ele2 * math.cos(vel_norm)
    #
    #     inside_sqrt = turning_radius**2 - (turning_radius - x_tmp)**2
    #     inside_sqrt = max(inside_sqrt, 0.0)
    #
    #     tan_pp = math.tan(vel_norm)
    #     if abs(tan_pp) < 1e-8:
    #         d_tmp = float("inf")
    #     else:
    #         d_tmp = math.sqrt(inside_sqrt) + x_tmp / tan_pp
    #
    #     if delta_ele2 < 0:
    #         k_rate_distance = 0.0
    #         continue
    #
    #     if dis_obstacle > v_range:
    #         dis_obstacle = v_range
    #
    #     d_need_to_AO = (d_tmp + delta_ele2 * math.sin(-vel_norm)) * factor_safety
    #
    #     if d_need_to_AO > dis_obstacle:
    #         return 1.0
    #
    # return 0.0








