# ==========================================
# 0. 基础配置（仅这一处需要改）
# ==========================================
# ① WiFi 指纹库
WIFI_FINGERPRINT_PATH = r"WiFi指纹库.xlsx"   # 放在项目根目录
# ② CAD 楼层图
FLOOR_PATHS = {
    '一层': r"一层.dxf",                           # 同上
    '二层': r"二层.dxf"
}
# =====================================================

import ezdxf
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString, MultiLineString, MultiPolygon
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid
import os
from heapq import heappop, heappush
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 解决库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

class CADRoomVisualizer:
    def __init__(self, floor_name=""):
        self.walls = []
        self.rooms = []
        self.measure_points = {}
        self.door_points = {}
        self.room_texts = []
        self.floor_name = floor_name
        self.wifi_points = {}
        self.target_counts = {
            '一层': {'numbers': 10, 'elevator': 2, 'stair': 3, 'corridor': 3, 'toilet': 2},
            '二层': {'numbers': 8, 'elevator': 2, 'stair': 3, 'corridor': 3, 'toilet': 1}
        }
        self.obstacles = None
        self.path = None
        self.shortest_path = None
        self.passable_areas = None
        self.all_paths = []

    # ---------- 以下全部为你的原有逻辑，仅删除硬盘路径 ----------
    def load_cad(self, cad_path):
        try:
            doc = ezdxf.readfile(cad_path)
        except Exception as e:
            print(f"读取CAD文件失败: {e}")
            return False
        msp = doc.modelspace()
        for ent in msp:
            if ent.dxftype() == 'LINE':
                x1, y1 = ent.dxf.start[0], ent.dxf.start[1]
                x2, y2 = ent.dxf.end[0], ent.dxf.end[1]
                self.walls.append(((x1, y1), (x2, y2)))
            elif ent.dxftype() == 'LWPOLYLINE':
                points = [(p[0], p[1]) for p in ent.get_points()]
                for i in range(len(points) - 1):
                    self.walls.append((points[i], points[i + 1]))
                if ent.closed and len(points) > 1:
                    self.walls.append((points[-1], points[0]))
            elif ent.dxftype() == 'TEXT':
                content = ent.dxf.text.strip()
                x, y = ent.dxf.insert[0], ent.dxf.insert[1]
                if not content:
                    continue
                if content.isdigit() and 1 <= int(content) <= 43:
                    self.measure_points[int(content)] = (x, y)
                elif content.isalpha() and len(content) == 1:
                    self.door_points[content] = (x, y)
                else:
                    if (len(content) == 3 and content.isdigit()) or content in ['电梯', '楼梯', '走廊', '厕所']:
                        self.room_texts.append((content, x, y))
        self._load_wifi_fingerprints()
        self._identify_rooms()
        self._count_room_types()
        self._build_obstacles()
        self._identify_passable_areas()
        return True

    def _load_wifi_fingerprints(self):
        if not os.path.exists(WIFI_FINGERPRINT_PATH):
            print(f"WiFi指纹库文件不存在: {WIFI_FINGERPRINT_PATH}")
            return
        try:
            df = pd.read_excel(WIFI_FINGERPRINT_PATH, engine='openpyxl', header=1)
            name_col, x_col, y_col = '固定点名', 'X坐标', 'Y坐标'
            if not all(c in df.columns for c in [name_col, x_col, y_col]):
                print(f"WiFi列名不匹配: {df.columns.tolist()}")
                return
            floor_mapping = {'1楼': '一层', '2楼': '二层', '一楼': '一层', '二楼': '二层'}
            for _, row in df.iterrows():
                try:
                    point_name = str(row[name_col]).strip()
                    x = float(row[x_col])
                    y = float(row[y_col])
                    point_floor = '一层'
                    for key, val in floor_mapping.items():
                        if key in point_name:
                            point_floor = val
                            break
                    if '101' in point_name:
                        point_floor = '一层'
                    if point_floor == self.floor_name:
                        scaled_x = x * 100
                        scaled_y = y * 100
                        self.wifi_points[point_name] = (scaled_x, scaled_y)
                except Exception as e:
                    print(f"解析WiFi点失败: {row.get(name_col, '未知')} - {e}")
                    continue
            print(f"{self.floor_name}加载WiFi点数量: {len(self.wifi_points)}")
        except Exception as e:
            print(f"加载WiFi失败: {e}")

    def _identify_passable_areas(self):
        passable_polygons = []
        for poly_coords, name in self.rooms:
            if name in ['走廊', '电梯', '楼梯']:
                poly = Polygon(poly_coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid:
                    passable_polygons.append(poly)
        self.passable_areas = unary_union(passable_polygons) if passable_polygons else MultiPolygon()

    def _build_obstacles(self):
        if not self.walls:
            self.obstacles = MultiPolygon()
            return
        lines = [LineString([p1, p2]) for p1, p2 in self.walls]
        merged_lines = unary_union(lines)
        if merged_lines.geom_type == 'LineString':
            merged_lines = MultiLineString([merged_lines])
        if merged_lines.geom_type == 'MultiLineString':
            polygons = list(polygonize(merged_lines))
            valid_polygons = []
            min_area = 6000 if self.floor_name == '一层' else 5000
            for poly in polygons:
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid and poly.area > min_area:
                    valid_polygons.append(poly)
            self.obstacles = unary_union(valid_polygons)

    def _count_room_types(self):
        counts = {'numbers': 0, 'elevator': 0, 'stair': 0, 'corridor': 0, 'toilet': 0}
        for _, name in self.rooms:
            if name.isdigit() and len(name) == 3:
                counts['numbers'] += 1
            elif name == '电梯':
                counts['elevator'] += 1
            elif name == '楼梯':
                counts['stair'] += 1
            elif name == '走廊':
                counts['corridor'] += 1
            elif name == '厕所':
                counts['toilet'] += 1
        self.room_counts = counts

    def _identify_rooms(self):
        if not self.walls:
            return
        lines = [LineString([p1, p2]) for p1, p2 in self.walls]
        merged_lines = unary_union(lines)
        if merged_lines.geom_type == 'LineString':
            merged_lines = MultiLineString([merged_lines])
        if merged_lines.geom_type == 'MultiLineString':
            polygons = list(polygonize(merged_lines))
            valid_polygons = []
            target_count = 20 if self.floor_name == '一层' else 17
            min_area = 3000 if self.floor_name == '一层' else 2500
            for poly in polygons:
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid and poly.area > min_area:
                    valid_polygons.append(poly)
            for poly in valid_polygons:
                inner_texts = []
                for content, x, y in self.room_texts:
                    if poly.contains(Point(x, y)):
                        inner_texts.append((content, Polygon(poly).centroid.distance(Point(x, y))))
                if inner_texts:
                    inner_texts.sort(key=lambda x: x[1])
                    room_name = inner_texts[0][0]
                    coords = np.array(poly.exterior.coords)[:-1]
                    self.rooms.append((coords, room_name))
        if self.floor_name == '一层' and '101' not in [n for _, n in self.rooms]:
            self._find_special_room('101')
        if len(self.rooms) < target_count:
            print(f"警告: {self.floor_name}只识别到{len(self.rooms)}个房间，目标{target_count}个")

    def _find_special_room(self, room_name):
        for content, x, y in self.room_texts:
            if content == room_name:
                buffer_poly = Point(x, y).buffer(800)
                coords = np.array(buffer_poly.exterior.coords)[:-1]
                self.rooms.append((coords, room_name))
                print(f"找到特殊房间: {room_name}")
                break

    def _a_star_algorithm(self, start, end):
        start_pt = Point(start)
        end_pt = Point(end)
        if not self.passable_areas.contains(start_pt):
            start = self._find_nearest_passable_point(start_pt)
            if start is None:
                return None
        if not self.passable_areas.contains(end_pt):
            end = self._find_nearest_passable_point(end_pt)
            if end is None:
                return None

        def heuristic(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        neighbors = [(-50, 0), (50, 0), (0, -50), (0, 50)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, end)}
        oheap = []
        heappush(oheap, (fscore[start], start))

        while oheap:
            _, current = heappop(oheap)
            if Point(current).distance(Point(end)) < 100:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path_length = sum(np.hypot(path[i][0] - path[i + 1][0], path[i][1] - path[i + 1][1]) for i in range(len(path) - 1))
                return {'path': path[::-1], 'length': path_length}
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in close_set or not self.passable_areas.contains(Point(neighbor)):
                    continue
                tentative_g_score = gscore[current] + heuristic(current, neighbor)
                if neighbor not in gscore or tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heappush(oheap, (fscore[neighbor], neighbor))
        return None

    # ==================== Flask 路由 ====================
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/get_floors')
    def get_floors():
        floors = []
        for floor_name in FLOOR_PATHS.keys():
            visualizer = CADRoomVisualizer(floor_name)
            if visualizer.load_cad(FLOOR_PATHS[floor_name]):
                measure_points = list(visualizer.measure_points.keys())
                floors.append({'name': floor_name, 'measure_points': [str(p) for p in measure_points]})
        return jsonify({'floors': floors})

    @app.route('/api/generate_map', methods=['POST'])
    def generate_map():
        data = request.json
        floor_name = data.get('floor')
        start_id = data.get('start_id')
        end_id = data.get('end_id')
        target_id = data.get('target_id')

        if not floor_name or floor_name not in FLOOR_PATHS:
            return jsonify({'error': '无效的楼层'}), 400

        visualizer = CADRoomVisualizer(floor_name)
        if not visualizer.load_cad(FLOOR_PATHS[floor_name]):
            return jsonify({'error': '加载CAD文件失败'}), 500

        start_point = None
        end_point = None
        target_point = None

        if start_id and start_id.isdigit():
            start_point = visualizer.measure_points.get(int(start_id))
        elif start_id in visualizer.wifi_points:
            start_point = visualizer.wifi_points.get(start_id)

        if end_id and end_id.isdigit():
            end_point = visualizer.measure_points.get(int(end_id))
        elif end_id in visualizer.wifi_points:
            end_point = visualizer.wifi_points.get(end_id)

        if target_id and target_id.isdigit():
            target_point = visualizer.measure_points.get(int(target_id))
        elif target_id in visualizer.wifi_points:
            target_point = visualizer.wifi_points.get(target_id)

        path_found = False
        if start_point and end_point:
            path_result = visualizer._a_star_algorithm(start_point, end_point)
            if path_result:
                visualizer.path = path_result['path']
                path_found = True

        fig, ax = plt.subplots(figsize=(15, 10))
        for (p1, p2) in visualizer.walls:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1)
        for coords, name in visualizer.rooms:
            color = 'yellow' if name in ['走廊', '电梯', '楼梯'] else 'white'
            ax.fill(coords[:, 0], coords[:, 1], color=color, edgecolor='black', alpha=0.3)
            if name:
                centroid = np.mean(coords, axis=0)
                ax.text(centroid[0], centroid[1], name, fontsize=8)
        for idx, (x, y) in visualizer.measure_points.items():
            ax.plot(x, y, 'bo', markersize=5)
            ax.text(x, y, str(idx), fontsize=8, color='blue')
        for name, (x, y) in visualizer.wifi_points.items():
            ax.plot(x, y, 'go', markersize=5)
            ax.text(x, y, name, fontsize=8, color='green')
        if visualizer.path:
            path_coords = np.array(visualizer.path)
            ax.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2)
        if start_point:
            ax.plot(start_point[0], start_point[1], 'ro', markersize=8)
            ax.text(start_point[0], start_point[1], '起点', fontsize=10, color='red')
        if end_point:
            ax.plot(end_point[0], end_point[1], 'mo', markersize=8)
            ax.text(end_point[0], end_point[1], '终点', fontsize=10, color='magenta')
        if target_point:
            ax.plot(target_point[0], target_point[1], 'co', markersize=8)
            ax.text(target_point[0], target_point[1], '目标', fontsize=10, color='cyan')
        ax.set_aspect('equal')
        plt.title(f'{floor_name}平面图与路径规划')
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        total_rooms = len(visualizer.rooms)
        stats = {
            'total_rooms': total_rooms,
            'room_counts': visualizer.room_counts,
            'wifi_points_count': len(visualizer.wifi_points),
            'path_found': path_found
        }
        return jsonify({'image': img_base64, 'stats': stats})

if __name__ == '__main__':
    app.run(debug=True)