#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCDCNFD字体设计 (PyQt5 专业重构版)
作者：跳舞的火公子
重构与注释：MCD
"""

# ==============================================================================
# SECTION 0: 核心依赖导入 - 纯 Qt 版本
# ==============================================================================

# --- Python 标准库 ---
import sys         # 用于系统相关的操作，如应用程序退出
import json        # 用于读写项目文件和配置 (JSON格式)
import os          # 用于处理文件和目录路径
import math        # 提供基础数学函数
import random      # 用于生成随机数 (例如，在AI模拟或笔画数估算中)
import threading   # (虽然Qt有自己的线程机制，但保留它可能对某些非GUI的后台逻辑有用)
import time        # 用于处理时间相关操作 (如时间戳)
import sqlite3     # 用于与字符元数据数据库进行交互
import unicodedata # 用于处理Unicode字符属性 (如获取类别)
import uuid        # 用于生成唯一的ID (例如，为部件和实例)
import copy        # 用于创建对象的深拷贝 (例如，在撤销/重做系统中)
import re          # 用于正则表达式操作 (例如，解析SVG路径)
import platform    # 用于检测操作系统以选择合适的默认字体
from datetime import datetime # 用于处理日期和时间戳
from typing import List, Tuple, Optional, Union, Dict, Any, Callable, Literal, Sequence, Iterable # 用于类型注解
from functools import lru_cache # 用于缓存计算结果，提高性能
from contextlib import contextmanager # 用于创建上下文管理器

# --- PyQt5 UI 框架 ---
# 这是构建整个图形用户界面的核心
from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QLineEdit, QComboBox, QFrame, QListView, QSplitter,
        QFileDialog, QMessageBox, QMenu, QColorDialog, QProgressDialog, QAction,
        QToolBar, QStatusBar, QDockWidget, QTabWidget, QSpinBox, QCheckBox,
        QSlider, QStyledItemDelegate, QStyle, QListWidget, QListWidgetItem,
        QAbstractItemView, QUndoStack, QUndoView, QUndoCommand, QSizePolicy, QScrollArea, # <-- 在这里添加 QUndoView
        QTextEdit, QGroupBox, QButtonGroup, QStyleOptionViewItem
    )
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QImage, QPixmap, QFont, QIcon,
    QPainterPath, QTransform, QPaintEvent, QPainterPathStroker
)
from PyQt5.QtCore import (
    Qt, QSize, QPoint, QPointF, QRect, QRectF, QAbstractListModel, QModelIndex,
    pyqtSignal, QObject, QRunnable, QThreadPool, QTimer, QSizeF
)

# --- 第三方核心逻辑库 ---
# 用于高性能的科学计算和矢量化操作
import numpy as np

# --- 可选的功能性依赖 ---
# 程序可以在没有这些库的情况下运行，但部分功能会受限

# 用于将设计好的字形编译成TTF字体文件
try:
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.misc.transform import Transform
    from fontTools.pens.basePen import BasePen
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    print("警告: fontTools 库未找到。TTF 字体导出功能将被禁用。")

# 用于自动获取汉字的拼音
try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("警告: pypinyin 库未找到。拼音相关功能将受限。")

# ==============================================================================
# SECTION 1: 数据模型与核心逻辑 (MODELS & CORE LOGIC)
#
# 重要提示:
# 以下所有类和函数均直接从原始的 MCDCNFD.py 文件中完整迁移而来。
# 这部分代码是UI无关的，构成了我们MVC架构中的“模型(Model)”层。
# 它们负责所有的数据结构、计算和业务逻辑，可以无缝地在新旧UI框架中使用。
# ==============================================================================

# --- 自定义类型定义 ---
Point = Tuple[float, float]
PointSequence = Union[List[Point], np.ndarray]
BezierSegment = Tuple[Point, Point, Point] # (P_start, C_control, P_end) for quadratic bezier
PathCommand = Union[
    Tuple[Literal['moveTo'], Point],
    Tuple[Literal['lineTo'], Point],
    Tuple[Literal['qCurveTo'], Point, Point],
    Tuple[Literal['curveTo'], Point, Point, Point],
    Tuple[Literal['closePath']],
]
# (x, y, pressure, timestamp, width_factor, speed_val)
HandwritingPointData = Tuple[float, float, float, float, float, float]
# 用于存储从外部文件加载的字符元数据
_EXTERNAL_HANZI_DATA: Dict[str, Dict[str, Any]] = {}


# --- 几何与数学辅助函数 ---

def _estimate_tangents(points: np.ndarray, is_closed: bool) -> np.ndarray:
    """
    [最终稳定版] 估算路径上每个点的切线向量。
    """
    num_points = len(points)
    if num_points < 2:
        return np.array([])
    
    tangents = np.zeros_like(points, dtype=float)

    if num_points == 2:
        vec = points[1] - points[0]
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            tangents[0] = tangents[1] = vec / norm
        return tangents

    for i in range(num_points):
        if i == 0:
            if is_closed:
                prev_p = points[num_points - 1]
                next_p = points[i + 1]
            else:
                prev_p = points[i]
                next_p = points[i + 1]
        elif i == num_points - 1:
            if is_closed:
                prev_p = points[i - 1]
                next_p = points[0]
            else:
                prev_p = points[i - 1]
                next_p = points[i]
        else:
            prev_p = points[i - 1]
            next_p = points[i + 1]
        
        vec = next_p - prev_p
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            tangents[i] = vec / norm
    return tangents



def _intersect_lines(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> Optional[np.ndarray]:
    """[内部辅助] 计算两条由点和方向向量定义的直线 (p1 + t*v1) 和 (p2 + u*v2) 的交点。"""
    # 构建线性方程组 Ax = b
    A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
    b = p2 - p1
    
    # 检查行列式，如果为零或接近零，则线平行或共线，无唯一交点
    determinant = np.linalg.det(A)
    if math.isclose(determinant, 0.0):
        return None
    
    try:
        # 求解参数 t
        t, _ = np.linalg.solve(A, b)
        # 计算交点坐标
        intersection_point = p1 + t * v1
        return intersection_point
    except np.linalg.LinAlgError:
        # 在极少数数值不稳定的情况下，求解可能失败
        return None



def _perpendicular_distance_squared_to_segment_vectorized(
    points_to_check: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
) -> np.ndarray:
    """
    [内部辅助] 矢量化计算一组点到一条线段的最短距离的平方。
    
    此函数正确处理了三种情况：
    1. 点的投影在线段内部：计算点到直线的垂直距离。
    2. 点的投影在线段外部、靠近起点：计算点到起点的距离。
    3. 点的投影在线段外部、靠近终点：计算点到终点的距离。

    Args:
        points_to_check (np.ndarray): 一个 (N, 2) 的数组，包含所有待检查的点。
        line_start (np.ndarray): 线段的起点坐标 (1, 2)。
        line_end (np.ndarray): 线段的终点坐标 (1, 2)。

    Returns:
        np.ndarray: 一个长度为 N 的一维数组，包含了每个点到线段距离的平方。
    """
    if points_to_check.shape[0] == 0:
        return np.array([])
    
    line_vector = line_end - line_start
    line_length_sq = np.sum(line_vector**2)

    # 处理线段起点和终点重合的特殊情况
    if line_length_sq == 0:
        return np.sum((points_to_check - line_start)**2, axis=1)

    # --- 核心几何计算 ---
    # 计算每个点相对于线段起点的向量
    point_to_start_vecs = points_to_check - line_start
    
    # 计算每个点在线段方向上的投影比例 t
    # t = dot(point_to_start, line_vector) / |line_vector|^2
    t = np.dot(point_to_start_vecs, line_vector) / line_length_sq
    
    # --- 分情况计算距离 ---
    distances_sq = np.zeros(points_to_check.shape[0])

    # Case 1: 投影落在线段内部 (0 <= t <= 1)
    # 使用二维叉乘的变体计算点到无限长直线的垂直距离的平方
    # dist^2 = (cross_product_z)^2 / |line_vector|^2
    cross_product_z = point_to_start_vecs[:, 0] * line_vector[1] - point_to_start_vecs[:, 1] * line_vector[0]
    distances_to_line_sq = cross_product_z**2 / line_length_sq
    
    idx_in_segment = (t >= 0) & (t <= 1)
    distances_sq[idx_in_segment] = distances_to_line_sq[idx_in_segment]

    # Case 2: 投影落在起点外侧 (t < 0)
    # 距离为点到线段起点的距离
    idx_t_lt_0 = t < 0
    if np.any(idx_t_lt_0):
        distances_sq[idx_t_lt_0] = np.sum((points_to_check[idx_t_lt_0] - line_start)**2, axis=1)

    # Case 3: 投影落在终点外侧 (t > 1)
    # 距离为点到线段终点的距离
    idx_t_gt_1 = t > 1
    if np.any(idx_t_gt_1):
        distances_sq[idx_t_gt_1] = np.sum((points_to_check[idx_t_gt_1] - line_end)**2, axis=1)
        
    return distances_sq

def ramer_douglas_peucker(points: PointSequence, epsilon: float) -> List[Point]:
    """
    使用 Ramer-Douglas-Peucker (RDP) 算法简化一个二维点序列。

    此算法通过递归（或迭代）地移除与线段近似在一条直线上的点，
    来减少曲线中的点数，同时保持其主要的几何形状。它是将手绘笔迹
    转换为平滑矢量曲线的关键预处理步骤。

    此版本为迭代实现，使用栈来避免递归深度限制，并进行了充分的
    输入验证和性能优化。

    Args:
        points (PointSequence): 
            一个包含 (x, y) 坐标元组的列表，或一个 (N, 2) 的 NumPy 数组。
        epsilon (float): 
            距离阈值。决定了简化的程度。算法将移除所有与
            近似线段距离小于此值的点。此值必须为非负数。

    Returns:
        List[Point]: 
            一个简化后的 (x, y) 坐标元组列表。列表中的点顺序
            与输入顺序保持一致。

    Raises:
        TypeError: 如果 `points` 不是列表或 NumPy 数组。
        ValueError: 如果 `points` 中的元素不是二维坐标，或者 `epsilon` 为负数。
    """
    # --- 1. 输入验证与数据准备 ---
    if not isinstance(points, (list, np.ndarray)):
        raise TypeError("输入点序列 `points` 必须是列表或 NumPy 数组。")
    
    if isinstance(points, list):
        # 从列表创建 NumPy 数组以进行高效计算
        np_points = np.array(points, dtype=float)
    else:
        # 确保 NumPy 数组是浮点类型
        np_points = points.astype(float)

    if np_points.ndim != 2 or np_points.shape[1] != 2:
        raise ValueError("输入点序列 `points` 中的每个元素必须是二维坐标 (x, y)。")
    
    num_points = len(np_points)
    
    # 如果点数过少，无法形成可简化的曲线，直接返回原始点
    if num_points < 3:
        return [tuple(p) for p in np_points]

    if epsilon < 0:
        raise ValueError("距离阈值 `epsilon` 必须为非负数。")
    
    # 性能优化：直接比较距离的平方，避免开方运算
    epsilon_sq = epsilon**2

    # --- 2. 算法初始化 ---
    # 使用一个布尔掩码来标记需要保留的点
    keep_mask = np.zeros(num_points, dtype=bool)
    keep_mask[0] = True
    keep_mask[num_points - 1] = True

    # 使用栈来实现迭代，栈中存储待处理线段的 (起始索引, 结束索引)
    stack = [(0, num_points - 1)]

    # --- 3. 主循环 ---
    while stack:
        start_idx, end_idx = stack.pop()

        # 如果线段之间没有中间点，则无需处理
        if end_idx - start_idx < 2:
            continue

        # 获取当前线段的所有中间点
        intermediate_indices = np.arange(start_idx + 1, end_idx)
        intermediate_points = np_points[intermediate_indices]

        if len(intermediate_points) == 0:
            continue
            
        # 调用辅助函数，计算所有中间点到线段 (start -> end) 的距离的平方
        distances_sq = _perpendicular_distance_squared_to_segment_vectorized(
            intermediate_points, np_points[start_idx], np_points[end_idx]
        )
        
        # 找到最大距离
        max_dist_sq = np.max(distances_sq)

        # --- 4. 核心判断与分割 ---
        if max_dist_sq > epsilon_sq:
            # 如果最大距离超过阈值，则这个最远点必须被保留
            idx_in_intermediate = np.argmax(distances_sq)
            farthest_idx = intermediate_indices[idx_in_intermediate]
            
            keep_mask[farthest_idx] = True

            # 将原线段以最远点为界，分割成两个新的子线段
            # 并将这两个子线段压入栈中，等待下一轮处理
            if farthest_idx - start_idx > 1:
                stack.append((start_idx, farthest_idx))
            if end_idx - farthest_idx > 1:
                stack.append((farthest_idx, end_idx))
    
    # --- 5. 返回结果 ---
    # 使用布尔掩码从原始点数组中筛选出所有被标记为保留的点
    # 并将它们转换回标准的元组列表格式
    return [tuple(p) for p in np_points[keep_mask]]


def fit_cubic_bezier_segments(
    points: Sequence[Point],
    smooth_factor: float = 0.25,
    is_closed: bool = False
) -> List[Tuple[Point, Point, Point, Point]]:
    """
    [新增][专业版] 将一系列锚点拟合成连续的三次贝塞尔曲线段。
    
    此算法通过估算每个锚点两侧的控制点来生成平滑的曲线。
    
    Args:
        points (Sequence[Point]): 锚点序列。
        smooth_factor (float): 平滑因子，控制控制点与锚点之间的距离。
                               数值越小，曲线越贴近直线。
        is_closed (bool): 路径是否闭合。
        
    Returns:
        List[Tuple[Point, Point, Point, Point]]: 
            一个包含 (P0, C1, C2, P1) 元组的列表，代表每一段三次贝塞尔曲线。
    """
    num_points = len(points)
    if num_points < 2:
        return []

    np_points = np.array(points, dtype=float)
    segments = []

    # 1. 计算所有线段的长度
    lengths = np.linalg.norm(np_points[1:] - np_points[:-1], axis=1)
    if is_closed:
        lengths = np.append(lengths, np.linalg.norm(np_points[0] - np_points[-1]))

    # 2. 估算所有锚点的切线
    tangents = _estimate_tangents(np_points, is_closed)

    # 3. 遍历每个线段，计算其两个控制点
    num_segments = num_points - 1 if not is_closed else num_points
    for i in range(num_segments):
        p0_idx = i
        p1_idx = (i + 1) % num_points

        p0 = tuple(np_points[p0_idx])
        p1 = tuple(np_points[p1_idx])

        t0 = tangents[p0_idx]
        t1 = tangents[p1_idx]

        # 根据线段长度和切线方向计算控制点
        # C1 是 P0 的出射控制点
        # C2 是 P1 的入射控制点
        dist = lengths[i] * smooth_factor
        c1 = tuple(np_points[p0_idx] + t0 * dist)
        c2 = tuple(np_points[p1_idx] - t1 * dist)

        segments.append((p0, c1, c2, p1))

    return segments

# --- 核心数据类 ---

class VectorPath:
    """
    [最终专业增强版 V3.1 - PyQt5 适配 & 完整实现] 矢量路径类

    用于表示、创建、操作、分析和渲染二维矢量路径。
    支持 SVG-like 的基本绘图命令、高级几何变换、精确边界框计算、
    路径优化、点类型查询以及与 SVG 路径数据的双向转换。

    此版本特别新增了 `to_qpainter_path()` 方法，以便能高效地在 PyQt5
    的 QPainter 中进行渲染，并完整实现了所有高级分析方法。
    """
    
    def __init__(self, commands: Optional[List[PathCommand]] = None):
        """
        初始化 VectorPath 对象。
        
        Args:
            commands (Optional[List[PathCommand]]): 一个可选的路径命令列表来初始化路径。
        """
        self._commands: List[PathCommand] = []
        self._current_point: Optional[Point] = None           # 跟踪绘图笔的逻辑位置
        self._start_point_of_subpath: Optional[Point] = None  # 跟踪当前子路径的起始点
        
        # 通过 _add_command 添加初始命令，以确保内部状态正确更新
        if commands:
            for cmd_tuple in commands:
                self._add_command(cmd_tuple)
        
        self._invalidate_caches()

    def _invalidate_caches(self):
        """
        使所有内部的、基于 lru_cache 的缓存失效。
        
        这个方法是确保数据一致性的关键。每当路径的内部命令列表
        (`self._commands`) 发生任何改变时，都必须调用此方法。
        它会清除所有依赖于路径数据的计算结果缓存（如边界框），
        强制下一次调用时重新计算。
        """
        # hasattr 检查增加了代码的健壮性
        if hasattr(self, 'get_bounds') and hasattr(self.get_bounds, 'cache_clear'):
            self.get_bounds.cache_clear()
        if hasattr(self, 'get_length') and hasattr(self.get_length, 'cache_clear'):
            self.get_length.cache_clear()
        if hasattr(self, 'get_all_points') and hasattr(self.get_all_points, 'cache_clear'):
            self.get_all_points.cache_clear()


    def _add_command(self, cmd_tuple: PathCommand):
        """[内部方法] 添加命令，更新内部状态，并使缓存失效。"""
        self._commands.append(cmd_tuple)
        
        # *** 关键步骤: 在修改数据后立即让缓存失效 ***
        self._invalidate_caches()
        
        cmd_name = cmd_tuple[0]
        if cmd_name == 'moveTo':
            self._current_point = cmd_tuple[1]
            self._start_point_of_subpath = cmd_tuple[1]
        elif cmd_name == 'lineTo':
            self._current_point = cmd_tuple[1]
        elif cmd_name == 'qCurveTo':
            self._current_point = cmd_tuple[2]
        elif cmd_name == 'curveTo':
            self._current_point = cmd_tuple[3]
        elif cmd_name == 'closePath':
            self._current_point = self._start_point_of_subpath
            self._start_point_of_subpath = None

    # --- 属性访问器 ---
    @property
    def commands(self) -> List[PathCommand]:
        """返回路径命令的列表（只读副本）。"""
        return list(self._commands)

    # --- 路径构建方法 (SVG-like API) ---
    def moveTo(self, x: float, y: float):
        self._add_command(('moveTo', (float(x), float(y))))
    
    def lineTo(self, x: float, y: float):
        if self._current_point is None: self.moveTo(x, y)
        self._add_command(('lineTo', (float(x), float(y))))
    
    def qCurveTo(self, cx: float, cy: float, x: float, y: float):
        if self._current_point is None: self.moveTo(x, y)
        self._add_command(('qCurveTo', (float(cx), float(cy)), (float(x), float(y))))
    
    def curveTo(self, cx1: float, cy1: float, cx2: float, cy2: float, x: float, y: float):
        if self._current_point is None: self.moveTo(x, y)
        self._add_command(('curveTo', (float(cx1), float(cy1)), (float(cx2), float(cy2)), (float(x), float(y))))
    
    def closePath(self):
        if self._start_point_of_subpath is not None:
            self._add_command(('closePath',))

    # --- 路径操作与分析 ---
    def is_empty(self) -> bool:
        """检查路径是否不包含任何命令。"""
        return not bool(self._commands)

    def get_subpaths(self) -> List['VectorPath']:
        """将路径分割成多个独立的子路径（由 moveTo 分隔）。"""
        subpaths: List['VectorPath'] = []
        current_subpath_commands: List[PathCommand] = []
        for cmd_tuple in self._commands:
            if cmd_tuple[0] == 'moveTo' and current_subpath_commands:
                subpaths.append(VectorPath(current_subpath_commands))
                current_subpath_commands = [cmd_tuple]
            else:
                current_subpath_commands.append(cmd_tuple)
        if current_subpath_commands:
            subpaths.append(VectorPath(current_subpath_commands))
        return subpaths

    @lru_cache(maxsize=1)
    def get_all_points(self) -> List[Point]:
        """获取路径中定义的所有点（锚点和控制点），结果将被缓存。"""
        all_points: List[Point] = []
        for cmd, *args in self._commands:
            if cmd in ('moveTo', 'lineTo'): all_points.append(args[0])
            elif cmd == 'qCurveTo': all_points.extend([args[0], args[1]])
            elif cmd == 'curveTo': all_points.extend([args[0], args[1], args[2]])
        return all_points

    @lru_cache(maxsize=1)
    def get_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        计算路径的精确边界框 (min_x, min_y, max_x, max_y)。
        此方法通过计算贝塞尔曲线的数学极值点，确保边界框完全包围路径。
        """
        if self.is_empty():
            return None

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        def update_bounds_with_point(p: Point):
            nonlocal min_x, min_y, max_x, max_y
            min_x = min(min_x, p[0])
            min_y = min(min_y, p[1])
            max_x = max(max_x, p[0])
            max_y = max(max_y, p[1])

        current_point: Optional[Point] = None
        start_point_of_subpath: Optional[Point] = None

        for cmd_tuple in self._commands:
            cmd_name = cmd_tuple[0]
            if cmd_name == 'moveTo':
                current_point = cmd_tuple[1]
                start_point_of_subpath = current_point
                update_bounds_with_point(current_point)
            elif current_point is None: continue
            elif cmd_name == 'lineTo':
                next_point = cmd_tuple[1]
                update_bounds_with_point(next_point)
                current_point = next_point
            elif cmd_name == 'qCurveTo':
                c1, p1 = cmd_tuple[1], cmd_tuple[2]
                for p in self._get_quadratic_bezier_extrema([current_point, c1, p1]):
                    update_bounds_with_point(p)
                current_point = p1
            elif cmd_name == 'curveTo':
                c1, c2, p1 = cmd_tuple[1], cmd_tuple[2], cmd_tuple[3]
                for p in self._get_cubic_bezier_extrema([current_point, c1, c2, p1]):
                    update_bounds_with_point(p)
                current_point = p1
            elif cmd_name == 'closePath':
                if start_point_of_subpath and current_point != start_point_of_subpath:
                    update_bounds_with_point(start_point_of_subpath)
                current_point = start_point_of_subpath
                start_point_of_subpath = None
        
        if min_x == float('inf'): return None
        return (min_x, min_y, max_x, max_y)

    def _get_quadratic_bezier_extrema(self, points: List[Point]) -> List[Point]:
        """[内部辅助] 计算二次贝塞尔曲线的极值点。"""
        p0, c1, p1 = points
        extrema = [p0, p1]
        for i in range(2):
            denominator = p0[i] - 2 * c1[i] + p1[i]
            if not math.isclose(denominator, 0.0):
                t = (p0[i] - c1[i]) / denominator
                if 0 < t < 1:
                    ex = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * c1[0] + t**2 * p1[0]
                    ey = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * c1[1] + t**2 * p1[1]
                    extrema.append((ex, ey))
        return extrema

    def _get_cubic_bezier_extrema(self, points: List[Point]) -> List[Point]:
        """[内部辅助] 计算三次贝塞尔曲线的极值点。"""
        p0, c1, c2, p1 = points
        extrema = [p0, p1]
        for i in range(2):
            a = 3 * (p1[i] - 3 * c2[i] + 3 * c1[i] - p0[i])
            b = 6 * (c2[i] - 2 * c1[i] + p0[i])
            c = 3 * (c1[i] - p0[i])
            if math.isclose(a, 0.0):
                if not math.isclose(b, 0.0):
                    t = -c / b
                    if 0 < t < 1: extrema.append(self._get_point_on_cubic_bezier(t, points))
                continue
            delta = b**2 - 4 * a * c
            if delta >= 0:
                sqrt_delta = math.sqrt(delta)
                t1, t2 = (-b + sqrt_delta) / (2 * a), (-b - sqrt_delta) / (2 * a)
                if 0 < t1 < 1: extrema.append(self._get_point_on_cubic_bezier(t1, points))
                if 0 < t2 < 1: extrema.append(self._get_point_on_cubic_bezier(t2, points))
        return extrema

    def _get_point_on_cubic_bezier(self, t: float, points: List[Point]) -> Point:
        """[内部辅助] 根据参数t计算三次贝塞尔曲线上的点坐标。"""
        p0, c1, c2, p1 = points
        omt = 1 - t
        x = omt**3*p0[0] + 3*omt**2*t*c1[0] + 3*omt*t**2*c2[0] + t**3*p1[0]
        y = omt**3*p0[1] + 3*omt**2*t*c1[1] + 3*omt*t**2*c2[1] + t**3*p1[1]
        return (x, y)

    @lru_cache(maxsize=1)
    def get_length(self, num_segments_per_bezier: int = 20) -> float:
        """
        通过将曲线分割为短直线段来估算路径的总长度。
        
        这是一个数值估算方法。它遍历路径中的每一段，对于直线段，
        它直接计算其欧几里得距离；对于贝塞尔曲线段，它会将其细分为
        `num_segments_per_bezier` 个微小的直线段，然后累加这些微小线段的长度。
        
        结果会被缓存以提高性能。

        Args:
            num_segments_per_bezier (int): 
                用于近似每条贝塞尔曲线的直线段数量。数值越高，结果越精确，
                但计算成本也越高。默认为 20。

        Returns:
            float: 估算出的路径总长度。
        """
        total_length = 0.0
        current_point: Optional[Point] = None
        start_point_of_subpath: Optional[Point] = None

        for cmd_tuple in self._commands:
            cmd_name = cmd_tuple[0]
            
            if cmd_name == 'moveTo':
                current_point = cmd_tuple[1]
                start_point_of_subpath = current_point
            
            elif current_point is None:
                continue

            elif cmd_name == 'lineTo':
                next_point = cmd_tuple[1]
                total_length += math.dist(current_point, next_point)
                current_point = next_point
                
            elif cmd_name == 'qCurveTo':
                c1, p1 = cmd_tuple[1], cmd_tuple[2]
                total_length += self._get_bezier_length(
                    current_point, c1, p1, 
                    num_segments=num_segments_per_bezier, bezier_type='quadratic'
                )
                current_point = p1
                
            elif cmd_name == 'curveTo':
                c1, c2, p1 = cmd_tuple[1], cmd_tuple[2], cmd_tuple[3]
                total_length += self._get_bezier_length(
                    current_point, c1, c2, p1,
                    num_segments=num_segments_per_bezier, bezier_type='cubic'
                )
                current_point = p1
                
            elif cmd_name == 'closePath':
                if start_point_of_subpath is not None and current_point != start_point_of_subpath:
                    # 计算闭合路径产生的隐式线段的长度
                    total_length += math.dist(current_point, start_point_of_subpath)
                current_point = start_point_of_subpath
                start_point_of_subpath = None
                
        return total_length

    def _get_bezier_length(self, p0: Point, *args: Point, num_segments: int, bezier_type: Literal['quadratic', 'cubic']) -> float:
        """
        [内部辅助] 通过线性近似法计算单段贝塞尔曲线的长度。
        """
        length = 0.0
        prev_point = p0
        
        for i in range(1, num_segments + 1):
            t = i / num_segments
            current_point = None
            
            if bezier_type == 'quadratic':
                c1, p1 = args[0], args[1]
                # 二次贝塞尔曲线公式: B(t) = (1-t)^2*P0 + 2(1-t)t*C1 + t^2*P1
                x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * c1[0] + t**2 * p1[0]
                y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * c1[1] + t**2 * p1[1]
                current_point = (x, y)
                
            elif bezier_type == 'cubic':
                c1, c2, p1 = args[0], args[1], args[2]
                # 三次贝塞尔曲线公式: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*C1 + 3(1-t)t^2*C2 + t^3*P1
                omt = 1 - t
                x = omt**3 * p0[0] + 3 * omt**2 * t * c1[0] + 3 * omt * t**2 * c2[0] + t**3 * p1[0]
                y = omt**3 * p0[1] + 3 * omt**2 * t * c1[1] + 3 * omt * t**2 * c2[1] + t**3 * p1[1]
                current_point = (x, y)

            if current_point:
                length += math.dist(prev_point, current_point)
                prev_point = current_point
                
        return length

    def concat(self, other_path: 'VectorPath') -> 'VectorPath':
        """将另一条路径连接到当前路径，返回一个新的 VectorPath 对象。"""
        new_commands = list(self._commands)
        new_commands.extend(other_path.commands)
        return VectorPath(new_commands)
    def transform(self, matrix: np.ndarray) -> 'VectorPath':
        """
        [新增的核心修正方法] 对路径进行几何变换，返回一个新的、变换后的路径实例。

        此方法是实现部件实例渲染和数据导出的关键。它遍历路径中的每一个
        命令，对命令中定义的所有坐标点应用一个标准的 3x3 仿射变换矩阵。

        Args:
            matrix (np.ndarray): 一个 3x3 的 NumPy 仿射变换矩阵。

        Returns:
            VectorPath: 一个包含所有变换后坐标点的新 VectorPath 对象。
        """
        # 如果路径为空，直接返回一个新的空路径，避免不必要的计算
        if self.is_empty():
            return VectorPath()

        def apply_transform(point: Point) -> Point:
            """[内部辅助函数] 将 3x3 仿射变换矩阵应用于单个二维点。"""
            # 1. 将 (x, y) 转换为齐次坐标 [x, y, 1]
            p_homogeneous = np.array([point[0], point[1], 1.0])
            
            # 2. 执行矩阵乘法
            #    (matrix @ p_homogeneous.T) 的结果是 [x', y', w']
            p_transformed = (matrix @ p_homogeneous.T)
            
            # 3. 从结果中提取变换后的 (x', y') 并返回
            return (p_transformed[0], p_transformed[1])

        new_commands: List[PathCommand] = []
        # 遍历当前路径的所有命令
        for cmd_tuple in self._commands:
            cmd_name = cmd_tuple[0]
            
            # 'closePath' 命令没有坐标点，直接原样添加
            if cmd_name == 'closePath':
                new_commands.append(cmd_tuple)
                continue
            
            # 提取命令中的所有坐标点 (例如 'qCurveTo' 有2个点)
            points_to_transform = cmd_tuple[1:]
            
            # 使用列表推导式对所有点应用变换
            transformed_points = [apply_transform(p) for p in points_to_transform]
            
            # 用变换后的点构建新的命令元组
            new_command_tuple = (cmd_name, *transformed_points)
            new_commands.append(new_command_tuple)
            
        # 使用包含所有新命令的列表创建一个全新的 VectorPath 实例并返回
        return VectorPath(new_commands)
    def copy(self) -> 'VectorPath':
        """创建当前路径的一个深拷贝。"""
        return VectorPath(copy.deepcopy(self._commands))

    @staticmethod
    def flatten_path(vector_path: 'VectorPath', segments_per_curve: int = 15) -> List[List[Point]]:
        """
        [静态辅助方法] 将一个 VectorPath 对象中的所有贝塞尔曲线扁平化（近似）为直线段列表。
        
        这个方法非常适合用于渲染，特别是当渲染环境不直接支持贝塞尔曲线时（例如 Pillow），
        或者需要对路径上的每个点进行操作（如计算可变宽度）时。

        工作流程:
        1. 遍历输入 VectorPath 的所有命令。
        2. 'moveTo' 开始一个新的子路径点列表。
        3. 'lineTo' 直接添加终点。
        4. 'qCurveTo' (二次贝塞尔) 通过在其参数 t=[0,1] 区间内进行多次插值，
           生成一系列点来近似曲线。
        5. 'closePath' 将子路径的起点添加到末尾，形成闭合形状。
        6. 最终返回一个包含所有子路径点列表的列表。

        Args:
            vector_path (VectorPath): 需要被扁平化的 VectorPath 对象。
            segments_per_curve (int): 用于近似每条二次贝塞尔曲线的直线段数量。
                                      数值越高，曲线越平滑，但计算量越大。

        Returns:
            List[List[Point]]: 
                一个列表的列表。每个子列表代表一个子路径（由moveTo或closePath分隔）
                的所有点坐标。例如: [[(x1,y1), (x2,y2)], [(x3,y3), ...]]
        """
        flattened_subpaths: List[List[Point]] = []
        current_flattened_path: List[Point] = []
        current_point: Optional[Point] = None
        start_point_of_subpath: Optional[Point] = None

        for cmd_tuple in vector_path.commands:
            cmd_name = cmd_tuple[0]
            
            if cmd_name == 'moveTo':
                # 如果当前子路径不为空，则先保存它
                if current_flattened_path:
                    flattened_subpaths.append(current_flattened_path)
                
                # 开始一个新的子路径
                current_point = cmd_tuple[1]
                current_flattened_path = [current_point]
                start_point_of_subpath = current_point
            
            elif current_point is None:
                # 如果路径不是以 moveTo 开始，则忽略无效命令
                continue

            elif cmd_name == 'lineTo':
                current_point = cmd_tuple[1]
                current_flattened_path.append(current_point)

            elif cmd_name == 'qCurveTo':
                control_point, end_point = cmd_tuple[1], cmd_tuple[2]
                
                # 通过插值生成一系列点来近似二次贝塞尔曲线
                for i in range(1, segments_per_curve + 1):
                    t = i / segments_per_curve
                    # 二次贝塞尔曲线的参数方程: B(t) = (1-t)^2*P0 + 2(1-t)t*C1 + t^2*P1
                    x = (1 - t)**2 * current_point[0] + 2 * (1 - t) * t * control_point[0] + t**2 * end_point[0]
                    y = (1 - t)**2 * current_point[1] + 2 * (1 - t) * t * control_point[1] + t**2 * end_point[1]
                    current_flattened_path.append((x, y))
                
                current_point = end_point
            
            # TODO: 可以添加对 'curveTo' (三次贝塞尔曲线) 的扁平化支持
            # elif cmd_name == 'curveTo':
            #     ...

            elif cmd_name == 'closePath':
                # 如果子路径的起点和终点不重合，则添加起点以闭合路径
                if start_point_of_subpath and current_point != start_point_of_subpath:
                    current_flattened_path.append(start_point_of_subpath)
                current_point = start_point_of_subpath
        
        # 将最后一个正在构建的子路径添加到结果中
        if current_flattened_path:
            flattened_subpaths.append(current_flattened_path)
            
        return flattened_subpaths

    # --- PyQt5 集成 ---
    def to_qpainter_path(self) -> QPainterPath:
        """[关键适配器方法] 将内部路径命令转换为 PyQt5 的 QPainterPath 对象。"""
        path = QPainterPath()
        if not self._commands: return path
        for cmd, *points in self._commands:
            if cmd == 'moveTo': path.moveTo(QPointF(*points[0]))
            elif cmd == 'lineTo': path.lineTo(QPointF(*points[0]))
            elif cmd == 'qCurveTo': path.quadTo(QPointF(*points[0]), QPointF(*points[1]))
            elif cmd == 'curveTo': path.cubicTo(QPointF(*points[0]), QPointF(*points[1]), QPointF(*points[2]))
            elif cmd == 'closePath': path.closeSubpath()
        return path

    # --- 序列化与反序列化 ---
    def to_svg_path_data(self, precision: int = 3) -> str:
        """
        将路径的内部命令转换为标准的 SVG 路径数据字符串。
        
        这个方法是实现矢量图形导出的核心功能之一。它遍历所有路径命令，
        并将它们翻译成 SVG <path> 元素的 'd' 属性所使用的文本格式。

        例如:
        [('moveTo', (10, 20)), ('lineTo', (30, 40))] 
        将被转换为:
        "M 10.000,20.000 L 30.000,40.000"

        Args:
            precision (int): 
                输出坐标值的小数位数。默认为 3。

        Returns:
            str: 格式化后的 SVG 路径数据字符串。
        """
        svg_data_parts: List[str] = []
        
        for cmd_tuple in self._commands:
            cmd_name = cmd_tuple[0]
            points = cmd_tuple[1:]
            
            if cmd_name == 'moveTo':
                x, y = points[0]
                svg_data_parts.append(f"M {x:.{precision}f} {y:.{precision}f}")
            
            elif cmd_name == 'lineTo':
                x, y = points[0]
                svg_data_parts.append(f"L {x:.{precision}f} {y:.{precision}f}")
                
            elif cmd_name == 'qCurveTo':
                # SVG 的二次贝塞尔曲线命令 'Q' 格式为: Q cx,cy x,y
                (cx, cy), (x, y) = points[0], points[1]
                svg_data_parts.append(f"Q {cx:.{precision}f} {cy:.{precision}f} {x:.{precision}f} {y:.{precision}f}")
                
            elif cmd_name == 'curveTo':
                # SVG 的三次贝塞尔曲线命令 'C' 格式为: C cx1,cy1 cx2,cy2 x,y
                (cx1, cy1), (cx2, cy2), (x, y) = points[0], points[1], points[2]
                svg_data_parts.append(f"C {cx1:.{precision}f} {cy1:.{precision}f} "
                                      f"{cx2:.{precision}f} {cy2:.{precision}f} "
                                      f"{x:.{precision}f} {y:.{precision}f}")
                                      
            elif cmd_name == 'closePath':
                # SVG 的闭合路径命令 'Z'
                svg_data_parts.append("Z")
                
        # 使用空格将所有命令片段连接成一个完整的字符串
        return " ".join(svg_data_parts)
    @staticmethod
    def from_quadratic_to_cubic_path(quad_path: 'VectorPath') -> 'VectorPath':
        """
        [静态辅助] 将一个只包含二次贝塞尔曲线的路径精确转换为三次贝塞尔曲线路径。
        二次曲线 (P0, P1, P2) 转换为三次曲线 (P0, C1, C2, P2) 的公式:
        C1 = P0 + 2/3 * (P1 - P0)
        C2 = P2 + 2/3 * (P1 - P2)
        """
        if quad_path.is_empty():
            return VectorPath()

        cubic_path = VectorPath()
        current_point = None

        for cmd, *points in quad_path.commands:
            if cmd == 'moveTo':
                current_point = points[0]
                cubic_path.moveTo(*current_point)
            
            elif cmd == 'lineTo':
                if current_point is None:
                    current_point = points[0]
                    cubic_path.moveTo(*current_point)
                else:
                    cubic_path.lineTo(*points[0])
                    current_point = points[0]

            elif cmd == 'qCurveTo':
                if current_point is None:
                    # 如果路径以qCurveTo开头，这是不规范的，但我们做兼容处理
                    # 假设起点就是控制点
                    current_point = points[0]
                    cubic_path.moveTo(*current_point)

                p0 = np.array(current_point)
                p1 = np.array(points[0]) # 二次曲线的控制点
                p2 = np.array(points[1]) # 二次曲线的终点

                # 应用数学公式进行转换
                c1 = p0 + 2/3 * (p1 - p0)
                c2 = p2 + 2/3 * (p1 - p2)

                cubic_path.curveTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
                current_point = tuple(p2)

            elif cmd == 'closePath':
                cubic_path.closePath()
                # 逻辑上，闭合后current_point会回到子路径起点，VectorPath内部会自动处理

        return cubic_path
    
    @classmethod
    def from_svg_path_data(cls, svg_data: str) -> 'VectorPath':
        """
        [类方法] 从一个标准的 SVG 路径数据字符串创建一个新的 VectorPath 实例。
        
        这个方法是一个功能强大的解析器，能够理解 SVG <path> 元素 'd' 属性的语法，
        包括绝对/相对坐标、多种命令以及隐式命令。

        这是实现从外部矢量图形（如 Adobe Illustrator, Inkscape）导入数据
        的关键功能。

        Args:
            svg_data (str): 包含路径数据的 SVG 字符串。
                            例如: "M 10 10 L 100 100 Q 150 10 200 100 Z"

        Returns:
            VectorPath: 一个根据 SVG 数据新创建的 VectorPath 对象。
            
        Raises:
            ValueError: 如果 SVG 数据格式不正确或包含不支持的命令。
        """
        path = cls()
        
        # 这个正则表达式用于将SVG路径字符串分解为命令和坐标数字的“令牌”列表
        # 它能处理逗号、空格、负数和科学计数法
        token_pattern = re.compile(r"([MLHVCSQAZmlhvcsqaz])|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
        tokens = [m.group(0) for m in token_pattern.finditer(svg_data) if m.group(0).strip()]

        i = 0
        current_cmd_char: Optional[str] = None
        
        # 内部辅助函数，用于从令牌流中安全地解析指定数量的坐标
        def _parse_coords(count: int) -> List[float]:
            nonlocal i
            coords: List[float] = []
            if i + count > len(tokens):
                raise ValueError(f"命令 '{current_cmd_char}' 期望 {count} 个坐标，但数据已结束。")
            for _ in range(count):
                try:
                    coords.append(float(tokens[i]))
                    i += 1
                except (ValueError, IndexError):
                    raise ValueError(f"命令 '{current_cmd_char}' 期望一个数字坐标，但得到了 '{tokens[i]}'。")
            return coords

        # --- 主解析循环 ---
        while i < len(tokens):
            token = tokens[i]
            if token.isalpha():
                current_cmd_char = token
                i += 1
            elif current_cmd_char is None:
                raise ValueError("SVG路径数据必须以一个命令开始。")
            
            # 确定是绝对坐标 (大写) 还是相对坐标 (小写)
            is_relative = current_cmd_char.islower()
            cmd_type = current_cmd_char.upper()
            
            # 获取当前逻辑笔尖的位置，用于计算相对坐标
            current_x, current_y = path._current_point or (0.0, 0.0)

            try:
                if cmd_type == 'M': # MoveTo
                    coords = _parse_coords(2)
                    target_x = current_x + coords[0] if is_relative else coords[0]
                    target_y = current_y + coords[1] if is_relative else coords[1]
                    path.moveTo(target_x, target_y)
                    # SVG 规范: M/m 后的多组坐标被视为隐式的 L/l
                    current_cmd_char = 'l' if is_relative else 'L'

                elif cmd_type == 'L': # LineTo
                    coords = _parse_coords(2)
                    target_x = current_x + coords[0] if is_relative else coords[0]
                    target_y = current_y + coords[1] if is_relative else coords[1]
                    path.lineTo(target_x, target_y)

                elif cmd_type == 'H': # Horizontal LineTo
                    coords = _parse_coords(1)
                    target_x = current_x + coords[0] if is_relative else coords[0]
                    path.lineTo(target_x, current_y)

                elif cmd_type == 'V': # Vertical LineTo
                    coords = _parse_coords(1)
                    target_y = current_y + coords[0] if is_relative else coords[0]
                    path.lineTo(current_x, target_y)

                elif cmd_type == 'Q': # Quadratic Bezier
                    coords = _parse_coords(4)
                    cx = current_x + coords[0] if is_relative else coords[0]
                    cy = current_y + coords[1] if is_relative else coords[1]
                    x = current_x + coords[2] if is_relative else coords[2]
                    y = current_y + coords[3] if is_relative else coords[3]
                    path.qCurveTo(cx, cy, x, y)
                
                elif cmd_type == 'C': # Cubic Bezier
                    coords = _parse_coords(6)
                    cx1 = current_x + coords[0] if is_relative else coords[0]
                    cy1 = current_y + coords[1] if is_relative else coords[1]
                    cx2 = current_x + coords[2] if is_relative else coords[2]
                    cy2 = current_y + coords[3] if is_relative else coords[3]
                    x = current_x + coords[4] if is_relative else coords[4]
                    y = current_y + coords[5] if is_relative else coords[5]
                    path.curveTo(cx1, cy1, cx2, cy2, x, y)
                
                elif cmd_type == 'Z': # ClosePath
                    path.closePath()
                
                # TODO: 可以添加对 S/s (smooth cubic), T/t (smooth quadratic), A/a (arc) 的支持
                
                else:
                    raise ValueError(f"不支持的SVG路径命令: '{current_cmd_char}'")
            
            except ValueError as e:
                # 重新抛出异常并附加上下文信息
                raise ValueError(f"解析SVG路径时出错 (在令牌 {i} 附近): {e}") from e
        
        return path
    @classmethod
    def from_points(cls, points: List[Point]) -> 'VectorPath':
        """从一个点列表创建一个简单的由直线段组成的VectorPath。"""
        path = cls()
        if not points:
            return path
        path.moveTo(points[0][0], points[0][1])
        for p in points[1:]:
            path.lineTo(p[0], p[1])
        return path


    # --- Python 魔法方法 ---
    def __len__(self) -> int:
        """返回路径中命令的数量。"""
        return len(self._commands)
        
    def __repr__(self) -> str:
        """返回对象的字符串表示，用于调试。"""
        if not self._commands: return "VectorPath(empty)"
        return f"VectorPath(commands={len(self._commands)})"

    def __eq__(self, other: Any) -> bool:
        """判断两个 VectorPath 是否相等。"""
        if not isinstance(other, VectorPath): return NotImplemented
        return self._commands == other._commands

    def __hash__(self) -> int:
        """为路径生成哈希值。"""
        return hash(tuple(self._commands))



# ==============================================================================
# 在 SECTION 1 中，找到 HandwritingStroke 类并用以下完整代码替换
# ==============================================================================

class HandwritingStroke:
    """
    [最终专业增强版 V2.1 - 已修复内部不一致性] 表示单个手绘笔画的专业版类。

    此版本修复了因重构不彻底导致的 AttributeError，统一使用 _raw_points
    作为内部点数据的唯一真实来源，确保了类所有方法的一致性和健壮性。
    """

    def __init__(
        self,
        points: Optional[List[HandwritingPointData]] = None,
        stroke_type: Literal['normal', 'pen', 'brush', 'marker', 'pencil', 'eraser', 'calligraphy'] = 'normal',
        color: str = '#2d3748',
        rdp_epsilon: float = 1.0,
        bezier_tension: float = 0.5,
        bezier_strategy: Literal['midpoint_normal_offset', 'tangent_intersection'] = 'midpoint_normal_offset',
        is_closed: bool = False,
        # [核心新增] 新增图层属性的构造函数参数
        name: Optional[str] = None,
        is_visible: bool = True,
        is_locked: bool = False,
        opacity: float = 1.0
    ):
        """
        [V3.1 - 完整图层属性] 初始化 HandwritingStroke 对象。
        
        此版本将用户的原始手绘输入 (_raw_points) 与其可编辑的矢量表示 (vector_path)
        彻底分离，并添加了支持完整图层系统所需的 name, is_visible, is_locked, 
        和 opacity 属性。

        Args:
            points (Optional[List[HandwritingPointData]]): 
                笔画的原始点序列。这被视为不可变的“真理”。
            stroke_type (Literal): 笔画类型，可能影响渲染方式。
            color (str): 笔画的十六进制颜色字符串。
            rdp_epsilon (float): Ramer-Douglas-Peucker 算法的距离阈值。
            bezier_tension (float): 贝塞尔曲线拟合的张力因子。
            bezier_strategy (Literal): 贝塞尔拟合策略。
            is_closed (bool): 路径是否应被视为闭合。
            name (Optional[str]): 图层的自定义名称。
            is_visible (bool): 图层是否可见。
            is_locked (bool): 图层是否被锁定，防止编辑。
            opacity (float): 图层的不透明度 (0.0 到 1.0)。
        """
        # --- 核心数据 ---
        self._raw_points: List[HandwritingPointData] = list(points) if points else []
        
        # 笔画元数据
        self.stroke_type = stroke_type
        self.color = color

        # --- 矢量化参数 ---
        self.rdp_epsilon = rdp_epsilon
        self.bezier_tension = bezier_tension
        self.bezier_strategy = bezier_strategy
        self.is_closed = is_closed
        
        # --- [核心新增] 新的图层属性 ---
        # 如果在创建时未提供名称，则生成一个基于UUID的唯一默认名称，确保可识别
        self.name = name or f"笔画 {uuid.uuid4().hex[:6]}"
        self.is_visible = is_visible
        self.is_locked = is_locked
        self.opacity = max(0.0, min(1.0, opacity)) # 确保透明度在 0.0-1.0 范围内

        # --- 锚点类型数据 ---
        self.anchor_types: List[Literal['corner', 'smooth', 'asymmetric']] = []

        # --- 缓存系统 ---
        self._cached_vector_path: Optional[VectorPath] = None
        self._last_vectorization_params: Optional[Tuple] = None
        
        # --- 观察者模式相关 ---
        self._observers: List[Callable] = []

        # --- 初始化逻辑 ---
        # 如果在创建实例时就提供了点数据，则让 to_bezier_path 在首次被调用时
        # 惰性地生成矢量路径。此处的 pass 表示无需立即执行任何操作。
        if self._raw_points:
            pass
    
    def _invalidate_vectorization_cache(self):
        """使缓存的矢量化数据失效。"""
        self._cached_vector_path = None
        self._last_vectorization_params = None
    
    def _notify_observers(self):
        """通知所有注册的观察者笔画数据已更新。"""
        for callback in self._observers:
            callback()

    # --- 属性 ---
    @property
    def points(self) -> List[HandwritingPointData]:
        """返回笔画原始点序列的只读副本。"""
        # [核心修正] 从 _raw_points 返回数据
        return list(self._raw_points)

    # --- 点操作 ---
    def add_point(self, x: float, y: float, pressure: float = 1.0, timestamp: Optional[float] = None):
        """
        向笔画添加一个点。
        """
        if timestamp is None: timestamp = time.time()
        # 速度值等衍生数据应由一个单独的 `recalculate_metrics` 方法处理，此处简化
        new_point = (float(x), float(y), float(pressure), float(timestamp), float(pressure), 0.0)
        # [核心修正] 添加到 _raw_points
        self._raw_points.append(new_point)
        self._invalidate_vectorization_cache()
        self._notify_observers()

    def clear_points(self):
        """清除笔画中的所有点。"""
        # [核心修正] 清空 _raw_points
        self._raw_points.clear()
        self._invalidate_vectorization_cache()
        self._notify_observers()

    # --- 矢量化与拟合 ---
    def to_bezier_path(self) -> VectorPath:
        """
        [专业升级版 V2.0] 将手绘点转换为平滑的三次贝塞尔曲线路径。
        
        此版本调用新的 fit_cubic_bezier_segments 算法，为实现高级节点编辑打下基础。
        同时，在生成路径时，它会初始化或重置 anchor_types 列表。
        """
        # --- 1. 缓存检查 ---
        current_params = (self.rdp_epsilon, self.bezier_tension, self.is_closed)
        
        # 如果缓存有效，直接返回缓存结果
        if self._cached_vector_path is not None and self._last_vectorization_params == current_params:
            return self._cached_vector_path
        
        # --- 2. 重新计算路径 ---
        
        # a. 安全性检查 (使用 self._raw_points)
        if len(self._raw_points) < 2:
            self.anchor_types = [] # 确保在无效时清空
            return VectorPath()
        
        # b. 使用 RDP 算法简化点序列，得到锚点 (使用 self._raw_points)
        raw_xy_points = [(p[0], p[1]) for p in self._raw_points]
        simplified_points = ramer_douglas_peucker(raw_xy_points, self.rdp_epsilon)
        
        if len(simplified_points) < 2:
            self.anchor_types = [] # 确保在无效时清空
            return VectorPath()

        # c. 初始化/重置 anchor_types 列表
        if len(self.anchor_types) != len(simplified_points):
            self.anchor_types = ['smooth'] * len(simplified_points)
            if not self.is_closed and len(self.anchor_types) > 0:
                self.anchor_types[0] = 'corner'
                self.anchor_types[-1] = 'corner'
            
        # d. 调用三次贝塞尔拟合函数
        bezier_segments = fit_cubic_bezier_segments(
            simplified_points,
            smooth_factor=self.bezier_tension * 0.5, # 将 tension 映射为平滑因子
            is_closed=self.is_closed
        )
        
        # e. 从三次贝塞尔段构建 VectorPath 对象
        path = VectorPath()
        if bezier_segments:
            path.moveTo(bezier_segments[0][0][0], bezier_segments[0][0][1])
            for p0, c1, c2, p1 in bezier_segments:
                path.curveTo(c1[0], c1[1], c2[0], c2[1], p1[0], p1[1])
            if self.is_closed:
                path.closePath()

        # --- 3. 更新缓存 ---
        self._cached_vector_path = path
        self._last_vectorization_params = current_params
        
        return path

    # --- 几何变换 ---
    def transform(self, matrix: np.ndarray) -> 'HandwritingStroke':
        """
        对笔画进行几何变换，返回一个新的、变换后的笔画实例 (遵循不可变性)。
        """
        # [核心修正] 检查 self._raw_points
        if not self._raw_points:
            return self.copy()

        # [核心修正] 从 self._raw_points 提取坐标
        xy_coords = np.array([(p[0], p[1]) for p in self._raw_points])
        homogeneous_coords = np.hstack([xy_coords, np.ones((len(xy_coords), 1))])
        transformed_coords_homogeneous = (matrix @ homogeneous_coords.T).T
        transformed_coords = transformed_coords_homogeneous[:, :2]

        # 创建新的点列表，保留原始元数据
        new_points: List[HandwritingPointData] = [
            (
                transformed_coords[i, 0], # 新 x
                transformed_coords[i, 1], # 新 y
                original_point[2], # 压力
                original_point[3], # 时间戳
                original_point[4], # 宽度因子
                original_point[5]  # 速度
            )
            for i, original_point in enumerate(self._raw_points) # [核心修正] 遍历 self._raw_points
        ]
        
        # 返回一个新的 HandwritingStroke 实例
        new_stroke = self.copy()
        # [核心修正] 更新新笔画的 _raw_points
        new_stroke._raw_points = new_points
        new_stroke._invalidate_vectorization_cache()
        return new_stroke

    # --- 序列化与通用方法 ---
    def copy(self) -> 'HandwritingStroke':
        """创建当前笔画的一个深拷贝。"""
        new_stroke = HandwritingStroke(
            points=copy.deepcopy(self._raw_points),
            stroke_type=self.stroke_type,
            color=self.color,
            rdp_epsilon=self.rdp_epsilon,
            bezier_tension=self.bezier_tension,
            bezier_strategy=self.bezier_strategy,
            is_closed=self.is_closed,
            # [核心新增] 复制新的图层属性
            name=self.name,
            is_visible=self.is_visible,
            is_locked=self.is_locked,
            opacity=self.opacity
        )
        
        new_stroke.anchor_types = self.anchor_types[:]
        new_stroke._cached_vector_path = self._cached_vector_path
        new_stroke._last_vectorization_params = self._last_vectorization_params
        
        return new_stroke

    def to_dict(self) -> Dict[str, Any]:
        """将笔画对象序列化为字典。"""
        return {
            'points': self._raw_points,
            'stroke_type': self.stroke_type,
            'color': self.color,
            'rdp_epsilon': self.rdp_epsilon,
            'bezier_tension': self.bezier_tension,
            'bezier_strategy': self.bezier_strategy,
            'is_closed': self.is_closed,
            'anchor_types': self.anchor_types,
            # [核心新增] 保存新的图层属性
            'name': self.name,
            'is_visible': self.is_visible,
            'is_locked': self.is_locked,
            'opacity': self.opacity,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HandwritingStroke':
        """从字典反序列化创建 HandwritingStroke 对象。"""
        points_raw = data.get('points', [])
        processed_points = []
        for p_tuple in points_raw:
            if len(p_tuple) == 6:
                processed_points.append(p_tuple)
            else:
                p = list(p_tuple)
                x, y, pressure, timestamp = p[0], p[1], p[2], p[3]
                width_factor = p[4] if len(p) >= 5 else pressure
                speed_val = 0.0
                processed_points.append((x, y, pressure, timestamp, width_factor, speed_val))
        
        instance = cls(
            points=processed_points,
            stroke_type=data.get('stroke_type', 'normal'),
            color=data.get('color', '#2d3748'),
            rdp_epsilon=data.get('rdp_epsilon', 1.0),
            bezier_tension=data.get('bezier_tension', 0.5),
            bezier_strategy=data.get('bezier_strategy', 'midpoint_normal_offset'),
            is_closed=data.get('is_closed', False),
            # [核心新增] 加载新的图层属性，并提供默认值以兼容旧项目文件
            name=data.get('name', None),
            is_visible=data.get('is_visible', True),
            is_locked=data.get('is_locked', False),
            opacity=data.get('opacity', 1.0)
        )
        
        instance.anchor_types = data.get('anchor_types', [])
        return instance

    @classmethod
    def from_vector_path(cls, vector_path: VectorPath, color: str = '#2d3748') -> 'HandwritingStroke':
        """
        [类方法][最终精准版 V3.1 - 已修复] 从一个 VectorPath 对象创建 HandwritingStroke。
        此版本修复了对已是三次曲线路径（如圆形工具）的处理逻辑。
        """
        instance = cls(color=color)
        if vector_path.is_empty():
            return instance

        # 1. [核心修正] 检查路径是否包含三次曲线命令。
        #    如果路径本身就是由 `curveTo` (三次曲线) 构成的（例如来自圆形工具），
        #    则无需进行二次到三次的转换，直接使用即可。
        #    否则，才执行转换，以兼容旧的二次曲线字形数据。
        has_cubic_curves = any(cmd[0] == 'curveTo' for cmd in vector_path.commands)
        if has_cubic_curves:
            cubic_path = vector_path  # 直接使用，不转换
        else:
            cubic_path = VectorPath.from_quadratic_to_cubic_path(vector_path) # 保持对二次曲线的兼容

        # 2. 从生成的三次路径中提取锚点...
        anchor_points_xy = []
        commands = cubic_path.commands
        if commands and commands[0][0] == 'moveTo':
            anchor_points_xy.append(commands[0][1])
            for cmd in commands[1:]:
                if cmd[0] == 'curveTo':
                    anchor_points_xy.append(cmd[3])
                elif cmd[0] == 'lineTo':
                    anchor_points_xy.append(cmd[1])

        now = time.time()
        instance._raw_points = [
            (x, y, 1.0, now, 1.0, 0.0) for x, y in anchor_points_xy
        ]
        
        # 3. [核心] 将精确转换后的三次路径直接注入缓存
        instance._cached_vector_path = cubic_path
        
        # 4. 同步缓存参数，防止缓存失效
        instance._last_vectorization_params = (
            instance.rdp_epsilon, 
            instance.bezier_tension, 
            instance.is_closed
        )

        return instance

    def get_preview_image(self, size: int, bg_color: QColor = Qt.transparent) -> QPixmap:
        """
        [纯Qt版] 生成笔画的预览图像 (QPixmap)。
        """
        pixmap = QPixmap(size, size)
        pixmap.fill(bg_color)
        
        # [核心修正] 检查 self._raw_points
        if len(self._raw_points) < 2:
            return pixmap
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        vector_path = self.to_bezier_path()
        bounds = vector_path.get_bounds()
        
        if not bounds:
            painter.end()
            return pixmap

        min_x, min_y, max_x, max_y = bounds
        content_w = max_x - min_x
        content_h = max_y - min_y

        if content_w < 1e-6 and content_h < 1e-6:
            painter.end()
            return pixmap

        border_margin = size * 0.1
        available_size = size - 2 * border_margin
        scale = available_size / max(content_w, content_h, 1.0)
        
        transform = QTransform()
        transform.translate((size - content_w * scale) / 2 - min_x * scale,
                            (size - content_h * scale) / 2 - min_y * scale)
        transform.scale(scale, scale)
        
        qpainter_path = vector_path.to_qpainter_path()
        final_path = transform.map(qpainter_path)
        
        pen = QPen(QColor(self.color), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(final_path)
        
        painter.end()
        return pixmap

    # --- Python 魔法方法 ---
    def __len__(self) -> int:
        """返回笔画中点的数量。"""
        # [核心修正]
        return len(self._raw_points)
        
    def __repr__(self) -> str:
        """返回对象的字符串表示，用于调试。"""
        return f"HandwritingStroke(points={len(self)}, color='{self.color}')"

class FontComponent:
    """
    [最终专业增强版 V4.0 - UI无关] 字体部件类。

    这是一个高度封装、可复用、带有元数据和变换信息的专业级字体部件。
    它被设计为独立的、不可变的设计资产，并为高性能渲染和未来的高级功能
    （如组件链接）做好了准备。

    此类的核心设计原则是“不可变性”：所有修改操作（如 transform, add_stroke）
    都会返回一个新的 FontComponent 实例，而不是在原地修改，这使得状态管理
    更加安全和可预测。
    """
    def __init__(self,
                 name: str,
                 strokes: Optional[List[HandwritingStroke]] = None,
                 component_uuid: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 origin: Point = (0.0, 0.0)):
        """
        初始化 FontComponent 对象。

        Args:
            name (str): 部件的名称，用于在UI中识别。
            strokes (Optional[List[HandwritingStroke]]): 构成此部件的笔画列表。
            component_uuid (Optional[str]): 部件的唯一标识符。如果为None，则会自动生成。
            metadata (Optional[Dict[str, Any]]): 存储额外信息的字典，如标签、描述等。
            origin (Point): 部件的逻辑原点，用于对齐和变换。
        """
        self.name: str = name
        self.uuid: str = component_uuid or str(uuid.uuid4())
        
        self.metadata: Dict[str, Any] = metadata or {
            'tags': [], 'category': '未分类', 'description': '',
            'created_at': datetime.now().isoformat()
        }
        
        self.origin: Point = origin
        # 笔画列表是深拷贝的，确保部件的独立性
        self._strokes: List[HandwritingStroke] = [s.copy() for s in (strokes or [])]
        self.timestamp: str = datetime.now().isoformat()
        
        # --- 缓存系统 ---
        self._bounds_cache: Optional[Tuple[float, float, float, float]] = None
        self._vector_path_cache: Optional[VectorPath] = None

    # --- 核心属性 ---
    @property
    def strokes(self) -> List[HandwritingStroke]:
        """返回部件包含的笔画列表的只读副本。"""
        return list(self._strokes)

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        返回部件的联合边界框 (min_x, min_y, max_x, max_y)。
        结果会被缓存以提高性能。
        """
        if self._bounds_cache is None:
            self._recalculate_bounds()
        return self._bounds_cache

    # --- 缓存与内部状态管理 ---
    def _invalidate_caches(self):
        """使所有内部缓存（如边界框、矢量路径）失效。"""
        self._bounds_cache = None
        self._vector_path_cache = None

    def _recalculate_bounds(self):
        """[内部方法] 根据所有笔画的原始点重新计算部件的联合边界框。"""
        if not self._strokes:
            self._bounds_cache = None
            return
        
        all_points = [p_data[:2] for s in self._strokes for p_data in s.points]
        if not all_points:
            self._bounds_cache = None
            return
            
        np_points = np.array(all_points)
        min_c = np.min(np_points, axis=0)
        max_c = np.max(np_points, axis=0)
        self._bounds_cache = (min_c[0], min_c[1], max_c[0], max_c[1])

    # --- 向量化 ---
    def to_vector_path(self) -> VectorPath:
        """将部件的所有笔画转换为一个单一的、缓存的 VectorPath 对象。"""
        if self._vector_path_cache is None:
            combined_path = VectorPath()
            for stroke in self._strokes:
                stroke_path = stroke.to_bezier_path()
                combined_path = combined_path.concat(stroke_path)
            self._vector_path_cache = combined_path
        return self._vector_path_cache

    # --- 修改操作 (返回新实例，遵循不可变性原则) ---
    def _clone_with_changes(self, **kwargs) -> 'FontComponent':
        """[内部] 创建当前实例的副本，并用kwargs中的值更新其属性。"""
        props = {
            'name': self.name,
            'strokes': self._strokes, # 注意：这里传递的是引用，因为通常是替换整个列表
            'component_uuid': self.uuid,
            'metadata': self.metadata,
            'origin': self.origin
        }
        props.update(kwargs)
        
        new_instance = FontComponent(**props)
        # 确保元数据是深拷贝，避免新旧实例共享元数据字典
        new_instance.metadata = copy.deepcopy(props['metadata'])
        return new_instance

    def add_stroke(self, stroke: HandwritingStroke, at_index: Optional[int] = None) -> 'FontComponent':
        """向部件添加一个笔画，并返回一个包含新笔画的新部件实例。"""
        new_strokes = self._strokes[:] # 创建列表的浅拷贝
        if at_index is not None:
            new_strokes.insert(at_index, stroke.copy())
        else:
            new_strokes.append(stroke.copy())
        return self._clone_with_changes(strokes=new_strokes)

    def transform(self, matrix: np.ndarray) -> 'FontComponent':
        """对部件进行几何变换，返回一个新的、变换后的部件实例。"""
        transformed_strokes = [s.transform(matrix) for s in self._strokes]
        
        # 变换原点
        origin_np = np.array([*self.origin, 1])
        transformed_origin_np = (matrix @ origin_np.T).T
        transformed_origin = tuple(transformed_origin_np[:2])
        
        return self._clone_with_changes(strokes=transformed_strokes, origin=transformed_origin)
        
    def translate(self, dx: float, dy: float) -> 'FontComponent':
        """平移部件，返回一个新的、平移后的部件实例。"""
        matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)
        return self.transform(matrix)

    def scale(self, sx: float, sy: Optional[float] = None, center: Optional[Point] = None) -> 'FontComponent':
        """围绕一个中心点缩放部件，返回一个新的、缩放后的部件实例。"""
        if sy is None: sy = sx
        center = center or self.origin
        cx, cy = center
        
        # 构造变换矩阵：先移至原点，再缩放，再移回
        t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        t_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        
        transform_matrix = t_pos @ s @ t_neg
        return self.transform(transform_matrix)

    def rotate(self, angle_degrees: float, center: Optional[Point] = None) -> 'FontComponent':
        """围绕一个中心点旋转部件，返回一个新的、旋转后的部件实例。"""
        b = self.bounds
        if center is None:
            center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b else self.origin
        
        cx, cy = center
        angle_rad = math.radians(angle_degrees)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        t_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        
        transform_matrix = t_pos @ rot @ t_neg
        return self.transform(transform_matrix)
        
    # --- 元数据API ---
    def set_metadata(self, key: str, value: Any) -> 'FontComponent':
        """设置一个元数据项，并返回一个新的部件实例。"""
        new_metadata = copy.deepcopy(self.metadata)
        new_metadata[key] = value
        return self._clone_with_changes(metadata=new_metadata)

    def add_tag(self, tag: str) -> 'FontComponent':
        """添加一个标签，并返回一个新的部件实例。"""
        new_metadata = copy.deepcopy(self.metadata)
        if 'tags' not in new_metadata: new_metadata['tags'] = []
        if tag not in new_metadata['tags']: new_metadata['tags'].append(tag)
        return self._clone_with_changes(metadata=new_metadata)

    # --- 序列化与反序列化 ---
    def to_dict(self) -> Dict[str, Any]:
        """将部件对象序列化为字典，以便保存。"""
        return {
            'uuid': self.uuid,
            'name': self.name,
            'metadata': self.metadata,
            'origin': self.origin,
            'strokes': [s.to_dict() for s in self._strokes],
            'timestamp': self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FontComponent':
        """从字典反序列化创建部件对象。"""
        return cls(
            name=data.get('name', '未命名部件'),
            strokes=[HandwritingStroke.from_dict(s) for s in data.get('strokes', [])],
            component_uuid=data.get('uuid'),
            metadata=data.get('metadata'),
            origin=tuple(data.get('origin', (0.0, 0.0)))
        )

    # --- 预览生成 ---
    def get_preview_image(self, size: int = 128, **kwargs) -> QPixmap:
        """[纯Qt版] 生成部件的预览图像 (QPixmap)。"""
        bg_color_tuple = kwargs.get('bg_color', Qt.transparent)
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(bg_color_tuple))
        
        if not self._strokes: return pixmap

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        vector_path = self.to_vector_path()
        bounds = vector_path.get_bounds()

        if not bounds:
            painter.end()
            return pixmap
        
        # ... (与 HandwritingStroke 中 get_preview_image 相同的变换和绘制逻辑) ...
        min_x, min_y, max_x, max_y = bounds
        content_w, content_h = max_x-min_x, max_y-min_y
        if content_w < 1e-6 and content_h < 1e-6:
            painter.end()
            return pixmap
        
        border_margin = kwargs.get('border_margin', size * 0.1)
        available_size = size - 2 * border_margin
        scale = available_size / max(content_w, content_h, 1.0)
        
        transform = QTransform()
        transform.translate((size-content_w*scale)/2-min_x*scale, (size-content_h*scale)/2-min_y*scale)
        transform.scale(scale, scale)
        
        final_path = transform.map(vector_path.to_qpainter_path())
        
        pen = QPen(QColor("#2D3748"), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(final_path)
        
        painter.end()
        return pixmap

    # --- Python 魔法方法 ---
    def __repr__(self) -> str:
        """返回对象的详细字符串表示，适合调试。"""
        return f"FontComponent(name='{self.name}', uuid='{self.uuid[:8]}', strokes={len(self._strokes)})"

    def __len__(self) -> int:
        """返回部件中笔画的数量。"""
        return len(self._strokes)
        
    def __eq__(self, other: Any) -> bool:
        """判断两个部件是否相等，基于其唯一的UUID。"""
        if not isinstance(other, FontComponent): return NotImplemented
        return self.uuid == other.uuid
        
    def __hash__(self) -> int:
        """部件的哈希值基于其唯一的UUID，使其可以安全地用作字典键。"""
        return hash(self.uuid)


class ComponentInstance:
    """
    [最终专业增强版 V1.0 - UI无关] 部件实例类。

    表示一个被放置在字符设计中的 `FontComponent` 的实例。
    此类本身不包含笔画数据，而是通过 `component_uuid` 引用一个
    全局的 `FontComponent` 定义。

    它的核心职责是存储该实例特有的变换信息（位置、缩放、旋转），
    并能将这些信息计算成一个标准的仿射变换矩阵，以便于渲染。
    """
    def __init__(self,
                 component_uuid: str,
                 position: Point = (0.0, 0.0),
                 scale: Union[float, Point] = 1.0,
                 rotation: float = 0.0):
        """
        初始化 ComponentInstance 对象。

        Args:
            component_uuid (str): 所引用的原始 FontComponent 的 UUID。
            position (Point, optional): 实例在字符坐标系中的位置 (平移量)。默认为 (0.0, 0.0)。
            scale (Union[float, Point], optional): 实例的缩放因子。
                - 如果是 float，则为统一缩放。
                - 如果是 Point (sx, sy)，则为非均匀缩放。
                默认为 1.0。
            rotation (float, optional): 实例的旋转角度（单位：度）。默认为 0.0。
        """
        self.component_uuid: str = component_uuid
        self.position: Point = position
        
        # 确保 scale 总是 (sx, sy) 的元组格式
        self.scale: Point = (scale, scale) if isinstance(scale, (int, float)) else scale
        
        self.rotation: float = rotation
        
        # 每个实例自身也有一个唯一的ID，用于在UI中区分不同的实例（即使它们引用同一个部件）
        self.instance_uuid: str = str(uuid.uuid4())

    def get_transform_matrix(self) -> np.ndarray:
        """
        将实例的位置、旋转和缩放参数计算成一个 3x3 的仿射变换矩阵。
        
        变换顺序遵循标准的计算机图形学约定：先在部件的原点 (0,0) 进行缩放，
        然后旋转，最后平移到目标位置。矩阵乘法的顺序与变换顺序相反。
        
        Matrix = Translate(tx, ty) * Rotate(angle) * Scale(sx, sy)

        Returns:
            np.ndarray: 一个 3x3 的 NumPy 仿射变换矩阵。
        """
        tx, ty = self.position
        sx, sy = self.scale
        angle_rad = math.radians(self.rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # 定义各个变换的矩阵
        translate_mat = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
        rotate_mat = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=float)
        scale_mat = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)
        
        # 矩阵乘法顺序与变换顺序相反
        transform_matrix = translate_mat @ rotate_mat @ scale_mat
        
        return transform_matrix

    def to_dict(self) -> Dict[str, Any]:
        """将部件实例序列化为字典，以便保存。"""
        return {
            'component_uuid': self.component_uuid,
            'position': self.position,
            'scale': self.scale,
            'rotation': self.rotation,
            'instance_uuid': self.instance_uuid
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentInstance':
        """从字典反序列化创建部件实例对象。"""
        # 必须提供 component_uuid
        if 'component_uuid' not in data:
            raise ValueError("从字典创建 ComponentInstance 失败：缺少 'component_uuid'。")

        instance = cls(
            component_uuid=data['component_uuid'],
            position=tuple(data.get('position', (0.0, 0.0))),
            scale=data.get('scale', 1.0),
            rotation=data.get('rotation', 0.0)
        )
        # 兼容旧数据：如果保存的数据中没有 instance_uuid，则保留自动生成的ID
        instance.instance_uuid = data.get('instance_uuid', instance.instance_uuid)
        return instance

    # --- Python 魔法方法 ---
    def __repr__(self) -> str:
        """返回对象的字符串表示，用于调试。"""
        return (f"ComponentInstance(ref='{self.component_uuid[:8]}...', "
                f"pos={self.position}, scale={self.scale}, rot={self.rotation})")

    def __eq__(self, other: Any) -> bool:
        """判断两个实例是否相等，基于其唯一的实例UUID。"""
        if not isinstance(other, ComponentInstance): return NotImplemented
        return self.instance_uuid == other.instance_uuid

    def __hash__(self) -> int:
        """实例的哈希值基于其唯一的实例UUID。"""
        return hash(self.instance_uuid)


class FontChar:
    """
    [最终专业增强版 V2.0 - UI无关] 表示单个字体字符的统一数据模型。

    此类是字体设计的核心单元，采用分层设计理念，一个字符可以由
    手绘笔画和多个可复用的、经过变换的部件实例共同构成。

    核心特性:
    - 分层数据结构: 独立管理 `_strokes` 和 `_components` 列表。
    - 性能优化: 内置了脏标记和缓存机制，用于优化矢量路径和边界框的计算。
    - 数据管理器集成: 与 FontDataManager 关联，用于元数据查询和状态同步。
    - UI无关: 所有方法都操作于纯数据层面，不依赖任何UI框架。
    """
    def __init__(self,
                 char: str,
                 char_data: Dict[str, Any],
                 grid_size: int,
                 data_manager: 'FontDataManager'):
        """
        初始化 FontChar 对象。

        Args:
            char (str): 字符本身，例如 '永'。
            char_data (Dict[str, Any]): 从 FontDataManager 获取的字符静态元数据。
            grid_size (int): 设计网格的大小 (例如 64x64)。
            data_manager (FontDataManager): 对数据管理器的引用。
        """
        if not data_manager:
            raise ValueError("FontChar 必须使用一个有效的 FontDataManager 实例进行初始化。")

        self.char = char
        self.unicode_val = ord(char)
        
        # --- 分层数据结构 ---
        self._strokes: List[HandwritingStroke] = []      # 手绘笔画层
        self._components: List[ComponentInstance] = []   # 部件实例层
        
        # 关联元数据和数据管理器
        self.metadata = char_data
        self.data_manager = data_manager
        
        # 从元数据中提取常用属性
        self.category = self.metadata.get('category', '未知')
        self.pinyin = self.metadata.get('pinyin', '')
        self.radical = self.metadata.get('radical', '未知')
        self.stroke_count_db = self.metadata.get('stroke_count', 0)
        
        # 字体度量
        self.grid_size = grid_size
        self.advance_width = float(grid_size) # 字符前进宽度
        
        # 状态
        self.is_designed = self.metadata.get('is_designed', 0) == 1
        
        # --- 性能与状态管理 ---
        self._is_dirty: bool = True # 脏标记，用于缓存控制
        self._vector_path_cache: Optional[VectorPath] = None
        self._bounds_cache: Optional[Tuple[float, float, float, float]] = None

    def _mark_dirty(self):
        """将字符标记为已修改，使其所有缓存失效，并更新设计状态。"""
        self._is_dirty = True
        self._vector_path_cache = None
        self._bounds_cache = None
        self.update_design_status()

    # --- 笔画层操作 ---
    @property
    def strokes(self) -> List[HandwritingStroke]:
        """返回手绘笔画列表的只读副本。"""
        return list(self._strokes)

    @strokes.setter
    def strokes(self, new_strokes: List[HandwritingStroke]):
        """安全地设置全新的笔画列表，并标记为已修改。"""
        self._strokes = [s.copy() for s in new_strokes]
        self._mark_dirty()

    def add_stroke(self, stroke: HandwritingStroke):
        """添加一个手绘笔画，并标记为已修改。"""
        self._strokes.append(stroke.copy())
        self._mark_dirty()

    # --- 部件层操作 ---
    @property
    def components(self) -> List[ComponentInstance]:
        """返回部件实例列表的只读副本。"""
        return list(self._components)

    def add_component(self, component_instance: ComponentInstance):
        """添加一个部件实例，并标记为已修改。"""
        self._components.append(component_instance)
        self._mark_dirty()

    # --- 通用清理与状态更新 ---
    def clear_all_layers(self):
        """清除所有设计层（笔画和部件），并标记为已修改。"""
        self._strokes.clear()
        self._components.clear()
        self._mark_dirty()
        
    def update_design_status(self):
        """根据当前笔画和部件数量更新 `is_designed` 状态。"""
        new_status = bool(self._strokes or self._components)
        if self.is_designed != new_status:
            self.is_designed = new_status
            self.metadata['is_designed'] = 1 if self.is_designed else 0

    # --- 统一渲染与计算 ---
    def to_vector_path(self, all_components: Dict[str, FontComponent]) -> VectorPath:
        """
        [已修正] 将此字符的所有层（手绘笔画和变换后的部件）合并为一个统一的VectorPath。
        结果会被缓存以提高性能。

        此版本依赖于新实现的 VectorPath.transform 方法，能够正确地将部件实例的
        变换（位置、缩放、旋转）应用到其基础矢量路径上。

        Args:
            all_components (Dict[str, FontComponent]):
                一个包含所有可用部件的字典（以UUID为键），用于解析部件实例。
        """
        # 如果字符未被修改且缓存有效，则直接返回缓存结果，提高性能
        if not self._is_dirty and self._vector_path_cache is not None:
            return self._vector_path_cache

        combined_path = VectorPath()

        # --- 步骤 1: 合并所有手绘笔画的路径 ---
        for stroke in self._strokes:
            combined_path = combined_path.concat(stroke.to_bezier_path())
            
        # --- 步骤 2: 合并所有变换后的部件实例的路径 ---
        for instance in self._components:
            # 从全局部件库中查找部件的原始定义
            original_component = all_components.get(instance.component_uuid)
            
            if original_component:
                # a. 获取该实例独有的仿射变换矩阵
                transform_matrix = instance.get_transform_matrix()
                
                # b. 获取原始部件的矢量路径
                component_path = original_component.to_vector_path()
                
                # c. [核心修正] 调用新增的 transform 方法应用变换
                #    这一步现在可以正确工作了
                transformed_path = component_path.transform(transform_matrix)
                
                # d. 将变换后的部件路径合并到最终路径中
                combined_path = combined_path.concat(transformed_path)

        # --- 步骤 3: 更新缓存并返回结果 ---
        self._vector_path_cache = combined_path
        self._is_dirty = False # 重置脏标记
        return self._vector_path_cache

    def get_bounds(self, all_components: Dict[str, FontComponent]) -> Optional[Tuple[float, float, float, float]]:
        """计算包含所有变换后部件的联合最小边界框。"""
        if not self._is_dirty and self._bounds_cache is not None:
            return self._bounds_cache

        combined_path = self.to_vector_path(all_components)
        self._bounds_cache = combined_path.get_bounds()
        return self._bounds_cache

    def get_preview_image(self, all_components: Dict[str, FontComponent], size: int = 128, **kwargs) -> QPixmap:
        """
        [纯Qt版] 生成字符的预览图像 (QPixmap)，会正确渲染所有图层。
        
        这个方法是字符视觉化的主要入口点。它根据字符的设计状态
        执行不同的渲染策略：
        - 如果字符已设计 (`is_designed` 为 True)，它会调用内部的 
          `_render_to_image` 方法来渲染真实的笔画和部件。
        - 如果字符未设计 (`is_designed` 为 False)，它会使用系统字体
          绘制一个灰色的占位符，为用户提供清晰的视觉提示。

        Args:
            all_components (Dict[str, FontComponent]): 
                一个包含所有可用部件的字典，用于在渲染时解析部件实例。
            size (int): 
                生成的 QPixmap 的边长（像素）。
            **kwargs: 
                可选的渲染参数，例如:
                - bg_color (QColor): 背景颜色，默认为透明。
                - border_margin (float): 内容与图像边缘的边距比例。

        Returns:
            QPixmap: 一个包含了字符预览的 Qt 图像对象。
        """
        bg_color = kwargs.get('bg_color', Qt.transparent)
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(bg_color))

        if not self.is_designed:
            # --- 未设计字符的占位符渲染逻辑 ---
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 尝试使用一个常见的、优雅的中文字体
            try:
                # 字体大小根据预览图尺寸动态调整
                font = QFont("Microsoft YaHei UI", int(size * 0.7))
            except Exception:
                # 如果找不到指定字体，则回退到系统默认字体
                font = QFont()
                font.setPixelSize(int(size * 0.7))
            
            painter.setFont(font)
            painter.setPen(QColor(180, 180, 180)) # 使用柔和的灰色
            
            # 在 pixmap 的矩形区域内居中绘制字符文本
            painter.drawText(pixmap.rect(), Qt.AlignCenter, self.char)
            
            painter.end() # 结束绘制
            return pixmap

        # --- 如果已设计，则调用内部的真实渲染方法 ---
        return self._render_to_image(all_components, size, **kwargs)

    def _render_to_image(self, all_components: Dict[str, FontComponent], size: int, **kwargs) -> QPixmap:
        """
        [内部辅助][纯Qt版] 将字符的矢量内容渲染到 QPixmap 上。
        """
        bg_color = kwargs.get('bg_color', Qt.transparent)
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(bg_color))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        bounds = self.get_bounds(all_components)
        if not bounds:
            painter.end()
            return pixmap
            
        min_x, min_y, max_x, max_y = bounds
        content_w = max_x - min_x
        content_h = max_y - min_y
        
        if content_w < 1e-6 and content_h < 1e-6:
            painter.end()
            return pixmap

        border_margin = kwargs.get('border_margin', size * 0.1)
        available_size = size - 2 * border_margin
        scale = available_size / max(content_w, content_h, 1.0)
        
        # 创建一个变换矩阵，将设计坐标映射到 pixmap 坐标
        transform = QTransform()
        transform.translate((size - content_w * scale) / 2 - min_x * scale,
                            (size - content_h * scale) / 2 - min_y * scale)
        transform.scale(scale, scale)

        # 渲染所有手绘笔画
        for stroke in self._strokes:
            path = transform.map(stroke.to_bezier_path().to_qpainter_path())
            pen = QPen(QColor(stroke.color), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush) # 路径不填充
            painter.drawPath(path)
        
        # 渲染所有部件实例
        for instance in self._components:
            component = all_components.get(instance.component_uuid)
            if not component: continue
            
            # 获取部件的矢量路径并应用实例变换
            instance_transform_matrix = instance.get_transform_matrix()
            for stroke in component.strokes:
                 transformed_stroke = stroke.transform(instance_transform_matrix)
                 path = transform.map(transformed_stroke.to_bezier_path().to_qpainter_path())
                 pen = QPen(QColor(transformed_stroke.color), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                 painter.setPen(pen)
                 painter.setBrush(Qt.NoBrush)
                 painter.drawPath(path)
        
        painter.end()
        return pixmap

    # --- 序列化与反序列化 ---
    def to_dict(self) -> Dict[str, Any]:
        """将 FontChar 对象序列化为字典。"""
        self.update_design_status()
        return {
            'char': self.char,
            'strokes': [s.to_dict() for s in self._strokes],
            'components': [c.to_dict() for c in self._components],
            'advance_width': self.advance_width,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], data_manager: 'FontDataManager', grid_size: int) -> 'FontChar':
        """从字典反序列化创建 FontChar 对象。"""
        char = data['char']
        # 确保能从数据管理器获取元数据，如果不存在则动态创建一个
        char_data = data_manager.get_char_info(char) or data_manager._get_char_metadata(char)
        
        instance = cls(char, char_data, grid_size, data_manager)
        instance._strokes = [HandwritingStroke.from_dict(s) for s in data.get('strokes', [])]
        instance._components = [ComponentInstance.from_dict(c) for c in data.get('components', [])]
        instance.advance_width = data.get('advance_width', float(grid_size))
        
        instance._mark_dirty() # 标记为脏，以便在首次使用时重新计算缓存
        return instance

    # --- Python 魔法方法 ---
    def __repr__(self) -> str:
        """返回对象的详细字符串表示，适合调试。"""
        return (f"FontChar(char='{self.char}', strokes={len(self._strokes)}, "
                f"components={len(self._components)}, designed={self.is_designed})")


class FontDataManager:
    """
    [最终专业增强版 V3.0 - 元数据服务] 字体数据管理器

    此类的核心职责是作为应用程序的静态“字符元数据服务”。它负责管理
    一个基础字符数据库，该数据库包含了大量字符的静态信息（如拼音、部首、
    预估笔画数等），但不包含用户动态创建的设计数据。

    主要功能:
    - 初始化: 确保基础字符数据库文件存在，如果不存在则自动创建一个。
    - 加载: 从指定的数据库文件将所有字符元数据加载到内存中以供快速查询。
    - 查询: 提供API来获取单个字符的信息或根据条件搜索字符。
    
    此版本遵循单一职责原则，不再管理动态的项目状态（如 `is_designed`），
    这些状态现在由 `FontChar` 对象自身管理。
    """
    def __init__(self, db_path: str = "font_char_data.db"):
        """
        初始化数据管理器。

        Args:
            db_path (str): 基础字符元数据数据库的文件路径。
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.characters: Dict[str, Dict[str, Any]] = {}

    def initialize(self):
        """
        初始化数据管理器。
        检查基础数据库文件是否存在，如果不存在，则创建一个并填充初始数据。
        """
        if os.path.exists(self.db_path):
            print(f"基础数据库 '{self.db_path}' 已存在，无需创建。")
            return

        print(f"未找到基础数据库 '{self.db_path}'。正在创建并填充...")
        try:
            # 使用 check_same_thread=False 可以在多线程环境中安全地传递连接对象
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()

            self._create_sqlite_table(cursor)
            initial_data = self._generate_default_char_data()
            self._save_to_sqlite(initial_data)
            print("基础数据库创建并填充完成。")

        except sqlite3.Error as e:
            print(f"SQLite错误: 创建数据库 '{self.db_path}' 失败: {e}")
            raise RuntimeError(f"无法创建核心字符数据库: {e}") from e
        finally:
            self.close()

    def load_database_from_file(self, db_path: str):
        """
        从指定的 .db 文件加载一个全新的字符元数据集到内存。
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"指定的数据库文件不存在: {db_path}")

        try:
            self.close()  # 关闭任何现有连接
            self.db_path = db_path
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            
            self.characters.clear()
            
            cursor.execute("SELECT * FROM characters ORDER BY unicode_val ASC")
            for row in cursor.fetchall():
                self.characters[row['char']] = dict(row)
            
            print(f"成功从数据库 '{self.db_path}' 加载 {len(self.characters)} 个字符元数据。")

        except sqlite3.Error as e:
            self.close()
            raise RuntimeError(f"无法加载指定的字符数据库: {e}") from e

    def _create_sqlite_table(self, cursor: sqlite3.Cursor):
        """创建 characters 表结构。"""
        cursor.execute('''
            CREATE TABLE characters (
                char TEXT PRIMARY KEY NOT NULL,
                unicode_val INTEGER NOT NULL,
                category TEXT,
                frequency INTEGER,
                pinyin TEXT,
                meaning TEXT,
                radical TEXT,
                stroke_count INTEGER,
                is_designed INTEGER DEFAULT 0,
                created_date TEXT,
                modified_date TEXT
            )
        ''')
        cursor.execute("CREATE INDEX idx_category ON characters (category);")
        cursor.execute("CREATE UNIQUE INDEX idx_unicode_val ON characters (unicode_val);")
        self.conn.commit()

    def _save_to_sqlite(self, data_dict: Dict[str, Dict[str, Any]]):
        """将内存中的数据字典一次性保存到SQLite数据库。"""
        if not self.conn or not data_dict: return
        try:
            cursor = self.conn.cursor()
            sample_data = next(iter(data_dict.values()))
            columns = list(sample_data.keys())
            placeholders = ', '.join(['?'] * len(columns))
            column_names = ', '.join(columns)
            
            data_to_save = [
                tuple(char_data.get(k) for k in columns) 
                for char_data in data_dict.values()
            ]
            
            cursor.executemany(f"INSERT OR REPLACE INTO characters ({column_names}) VALUES ({placeholders})", data_to_save)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite错误: 保存初始数据失败: {e}")
            self.conn.rollback()

    def get_all_chars_data(self) -> List[Dict[str, Any]]:
        """返回内存中所有字符元数据的列表。"""
        return list(self.characters.values())
    
    def get_char_info(self, char: str) -> Optional[Dict[str, Any]]:
        """获取单个字符的元数据信息。"""
        return self.characters.get(char)
        
    def search_chars(self, query: str = "", category: str = "全部") -> List[Dict[str, Any]]:
        """
        在内存中的元数据上执行搜索和过滤。
        此方法不关心 'is_designed' 状态，该过滤应由更高层的控制器处理。
        """
        results = []
        for char_data in self.characters.values():
            # 1. 过滤查询字符串 (不区分大小写)
            if query:
                query_lower = query.lower()
                if not any(query_lower in (str(val) or '').lower() for key, val in char_data.items() if key in ['char', 'pinyin', 'radical', 'meaning']):
                    continue
            
            # 2. 过滤分类
            if category != '全部' and char_data.get('category') != category:
                continue

            results.append(char_data)
        
        return results
    
    def close(self):
        """关闭数据库连接。"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("SQLite连接已关闭。")

    # --- 默认数据生成辅助函数 ---
    
    def _generate_default_char_data(self) -> Dict[str, Dict[str, Any]]:
        """生成并返回一个包含所有默认字符及其元数据的字典。"""
        common_hanzi = self._get_common_chinese_chars_internal(limit=3000)
        punctuation = list("，。？！；：""''「」（）《》〈〉﹃﹄﹁﹂「」[]{}()!?.,;:\"'-—~·《》〈〉")
        numbers = [str(i) for i in range(10)] + list('零一二三四五六七八九十百千万')
        english_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        other_symbols = ['+', '-', '*', '/', '=', '€', '$', '¥', '&', '@', '#', '%', '^', '_', '|', '\\', '<', '>']

        all_default_chars = list(dict.fromkeys(common_hanzi + punctuation + numbers + english_letters + other_symbols))
        
        default_data_dict = {}
        for char in all_default_chars:
            if char.strip():
                default_data_dict[char] = self._get_char_metadata(char)
        
        return default_data_dict

    def _get_common_chinese_chars_internal(self, limit: int = 3000) -> List[str]:
        """从内置源字符串中提取唯一的常用汉字。"""
        source_str = "的一是不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回住分爱老因很给名法间斯知世什二次使身者被高已亲其进此话常与活感见明问力理尔点文几定本公特做外孩相西果走将月分实发声车全信重三机工物气每并别真打太新比才便夫再书部水千眼等体制加电主界门利海内听表德少克代员许稳先口由死安写性马光白或者难望教命花结乐色更拉东神记处让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原像妈变通师立象数四失满战远士音轻目条呢病始深完今想求清王化空力业思切怎非找片罗钱南语元喜曾离飞科言干流欢约指合反题必该论交终林请医晚制球决传画保读运及则房早院苦火布品近产答星精视五连司巴奇管类朋且婚小夜青北队久乎越观落尽形影红爸百令周史识步希亚术留市半热送兴造谈容随演收首根讲整式取照办强石古华谊拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断禁城故历惊脸选包紧争叫建维绿层册巾股份歌姐纵幅晓亦酬技秀汗豆苏亿构贺谈虎骑粒逐毛驱蜜农桑团锦"
        unique_chars = list(dict.fromkeys(c for c in source_str if '\u4e00' <= c <= '\u9fff'))
        return unique_chars[:limit]

    def _get_char_metadata(self, char: str) -> Dict[str, Any]:
        """为单个字符生成元数据字典。"""
        ext_info = _EXTERNAL_HANZI_DATA.get(char, {})
        now_iso = datetime.now().isoformat()
        return {
            'char': char,
            'unicode_val': ord(char),
            'category': ext_info.get('category', self._infer_char_category(char)),
            'frequency': ext_info.get('freq', self._infer_char_frequency(char)),
            'pinyin': ext_info.get('pinyin', self._get_char_pinyin(char)),
            'meaning': ext_info.get('meaning', ''),
            'radical': ext_info.get('radical', self._infer_char_radical(char)),
            'stroke_count': ext_info.get('strokes', self._estimate_stroke_count(char)),
            'is_designed': 0,
            'created_date': now_iso,
            'modified_date': now_iso
        }

    def _infer_char_category(self, char: str) -> str:
        """根据字符推断其类别。"""
        if not isinstance(char, str) or len(char) != 1: return '无效字符'
        if '0' <= char <= '9': return '数字'
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z': return '字母'
        if '\u4e00' <= char <= '\u9fff': return '汉字'
        cat = unicodedata.category(char)
        if cat.startswith(('P', 'S')): return '符号'
        return '其他'

    def _infer_char_frequency(self, char: str) -> int:
        """根据内置列表推断字符使用频率等级。"""
        high_freq_chars = "的一是不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看"
        if char in high_freq_chars: return 1000
        if '\u4e00' <= char <= '\u9fff': return 100
        return 10

    def _infer_char_radical(self, char: str) -> str:
        """简化版的部首推断。"""
        return '未知'

    def _estimate_stroke_count(self, char: str) -> int:
        """简化版的笔画数估算。"""
        if not ('\u4e00' <= char <= '\u9fff'): return 1
        return random.randint(1, 15)

    def _get_char_pinyin(self, char: str) -> str:
        """获取字符的拼音，如果pypinyin可用。"""
        if PYPINYIN_AVAILABLE and '\u4e00' <= char <= '\u9fff':
            try:
                py_list = pinyin(char, style=Style.NORMAL)
                return ",".join(p[0] for p in py_list if p)
            except Exception as e:
                print(f"Pypinyin error for char '{char}': {e}")
                return ""
        return ""
# ==============================================================================
# SECTION 2: QT 特定模型与委托 (QT MODELS & DELEGATES)
#
# 这部分代码是连接纯Python数据模型和Qt视图的桥梁，是实现高性能UI的关键。
# 它们遵循Qt的Model/View架构，实现了关注点分离。
# ==============================================================================

class CharListModel(QAbstractListModel):
    """
    字符列表的Qt数据模型 (Model)。

    这是实现高性能虚拟化列表的核心组件，它遵循Qt的Model/View架构。
    
    职责:
    - 作为一个适配器，连接后端的纯Python数据列表 (List[FontChar]) 和前端的UI视图 (QListView)。
    - 向视图报告它管理了多少项数据。
    - 根据视图的请求，按需提供每一项的数据片段（例如，文本、工具提示、或整个数据对象）。
    - 当底层数据发生变化时，通知所有关联的视图进行刷新。

    它本身不关心数据如何被显示，只负责提供数据。
    """
    def __init__(self, parent=None):
        """
        初始化模型。
        
        Args:
            parent (QObject, optional): 父对象。默认为 None。
        """
        super().__init__(parent)
        self._chars: List[FontChar] = [] # 内部数据存储，一个 FontChar 对象的列表

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        [必要重写] 返回模型中的项目总数。
        
        QListView 会调用此方法来了解列表的总长度，以便正确设置滚动条
        和进行内部布局计算。

        Args:
            parent (QModelIndex): 对于一维列表模型，此参数通常不使用。

        Returns:
            int: 列表中的字符总数。
        """
        return len(self._chars)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """
        [必要重写] 根据指定的角色(role)为给定的索引(index)提供数据。
        
        这是模型/视图架构中最重要的函数。视图（或委托）会用不同的角色
        来请求不同类型的数据，以构建用户界面。

        Args:
            index (QModelIndex): 请求数据的项的索引，包含了行号和列号。
            role (int): 请求的数据类型。例如 Qt.DisplayRole 表示请求显示的文本。

        Returns:
            Any: 根据角色返回相应的数据，如果角色不支持或索引无效，则返回 None。
        """
        # --- 步骤 1: 安全性检查 ---
        # 确保索引是有效的，并且行号在我们的数据范围内。
        if not index.isValid() or not (0 <= index.row() < len(self._chars)):
            return None

        # --- 步骤 2: 获取对应的数据对象 ---
        char_obj: FontChar = self._chars[index.row()]
        
        # --- 步骤 3: 根据角色返回不同的数据 ---
        if role == Qt.DisplayRole:
            # `DisplayRole`: 提供默认显示的文本。
            return char_obj.char
        
        if role == Qt.UserRole:
            # `UserRole`: 这是一个非常实用的自定义角色，我们用它来传递整个 FontChar 对象。
            # 这样，控制器就可以从索引中轻松获取完整的数据对象，而不仅仅是字符串。
            return char_obj
            
        if role == Qt.ToolTipRole:
            # `ToolTipRole`: 提供当鼠标悬停在列表项上时显示的提示文本。
            pinyin = f" [{char_obj.pinyin}]" if char_obj.pinyin else ""
            status = '已设计' if char_obj.is_designed else '未设计'
            return f"{char_obj.char}{pinyin}\n状态: {status}\nUnicode: U+{char_obj.unicode_val:04X}"
        
        # 如果请求的角色不是我们处理的，则返回 None。
        return None

    def set_characters(self, chars: List[FontChar]):
        """
        用一个新的字符数据列表完全替换模型中的现有数据。
        
        这是更新模型的主要入口点。它会通知所有连接的视图，数据已经
        发生了根本性的变化，需要进行一次彻底的刷新。

        Args:
            chars (List[FontChar]): 新的 FontChar 对象列表。
        """
        # `beginResetModel()`: 告诉视图：“准备好，我要重置所有数据了！”
        self.beginResetModel()
        
        # 替换内部数据存储
        self._chars = chars
        
        # `endResetModel()`: 告诉视图：“数据重置完成，现在可以从头开始请求新数据了。”
        self.endResetModel()

    def get_char_obj_by_index(self, index: QModelIndex) -> Optional[FontChar]:
        """
        一个方便的辅助方法，用于从模型索引直接获取完整的 FontChar 对象。
        它内部利用了 `data(index, Qt.UserRole)`。

        Args:
            index (QModelIndex): 目标项的模型索引。

        Returns:
            Optional[FontChar]: 对应的 FontChar 对象，如果索引无效则返回 None。
        """
        if index.isValid():
            return self.data(index, Qt.UserRole)
        return None

class ReferenceImageWorker(QRunnable):
    """
    一个在后台生成参考底模图像的工作器。
    [最终修正版]：此版本接收 zhonggong_scale 参数，以确保生成的底图
    能与画布上的专业辅助线（特别是中宫）完美对齐。
    """
    class Signals(QObject):
        finished = pyqtSignal(QPixmap)

    def __init__(self, char: str, size: int, zhonggong_scale: float = 0.78):
        """
        初始化工作器。

        Args:
            char (str): 要渲染的字符。
            size (int): 生成的 QPixmap 的边长（像素）。
            zhonggong_scale (float): 
                中宫占整个字面框的比例。这是确保底图与辅助线
                对齐的关键参数。默认为 0.78。
        """
        super().__init__()
        self.char = char
        self.size = size
        self.zhonggong_scale = zhonggong_scale
        self.signals = self.Signals()

    def run(self):
        """
        [后台线程][已修复TypeError] 使用 QPainter 并根据“中宫”比例创建字符图像。
        
        此版本修复了将浮点数传递给 QRect.adjusted() 方法而导致的 TypeError。
        """
        try:
            # 1. 创建一个透明的画布
            pixmap = QPixmap(self.size, self.size)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # --- 根据中宫比例计算绘制区域 ---
            
            # 2. 计算中宫区域在 Pixmap 内部的边距（结果为浮点数）
            inset_float = self.size * (1 - self.zhonggong_scale) / 2.0
            
            # [核心修正] 将计算出的浮点数边距转换为整数
            inset = int(inset_float)

            # 3. 根据整数边距，从原始 pixmap.rect() 向内收缩，得到正确的绘制矩形
            draw_rect = pixmap.rect().adjusted(inset, inset, -inset, -inset)

            # 4. 选择字体，并使用 setPixelSize 精确控制字体在垂直方向撑满“中宫”区域
            try:
                font = QFont("Microsoft YaHei UI")
            except Exception:
                font = QFont()
            
            font.setPixelSize(int(draw_rect.height()))
            painter.setFont(font)

            # 5. 设置画笔颜色
            painter.setPen(QColor(180, 180, 180, 100))
            
            # 6. 在计算出的“中宫”矩形 (draw_rect) 中居中绘制文字
            painter.drawText(draw_rect, Qt.AlignCenter, self.char)
            
            painter.end()
            
            # 7. 任务完成，发射信号
            self.signals.finished.emit(pixmap)
        except Exception as e:
            print(f"为 '{self.char}' 生成参考底模时出错: {e}")

class ExportWorker(QRunnable):
    """
    一个可在后台线程池中运行的TTF导出任务。

    此类继承自 QRunnable，专门用于在后台执行耗时的字体编译工作，
    以避免阻塞主 UI 线程。它通过 Qt 的信号与槽机制与主线程进行安全的通信，
    报告进度、成功结果或错误信息。
    """
    
    class Signals(QObject):
        """
        定义此工作器可以发射的自定义信号。
        必须继承自 QObject 才能定义信号。
        """
        # 进度信号：(当前处理的索引, 总数, 当前字符名)
        progress = pyqtSignal(int, int, str)
        # 完成信号：(成功保存的文件路径)
        finished = pyqtSignal(str)
        # 错误信号：(错误信息字符串)
        error = pyqtSignal(str)

    def __init__(self, filename: str, font_chars: List['FontChar'], font_metadata: Dict, 
                 kerning_pairs: Dict, options: Dict):
        """
        初始化工作器。

        Args:
            filename (str): 目标 .ttf 文件的保存路径。
            font_chars (List[FontChar]): 所有待处理的 FontChar 对象列表。
            font_metadata (Dict): 包含字体元数据（如名称、度量）的字典。
            kerning_pairs (Dict): 包含字偶距信息的字典。
            options (Dict): 包含导出选项（如是否子集化）的字典。
        """
        super().__init__()
        self.signals = self.Signals()
        
        # --- 任务参数 ---
        self.filename = filename
        self.font_chars = font_chars
        self.font_metadata = font_metadata
        self.kerning_pairs = kerning_pairs
        self.options = options
        
        # --- 状态标志 ---
        self.is_cancelled = False

    def cancel(self):
        """
        [公共方法，由主线程调用] 设置取消标志，以请求中断任务。
        """
        print("后台导出任务被请求取消...")
        self.is_cancelled = True

    
    def _convert_path_to_pen(self, path: VectorPath, pen: BasePen, scale: float, y_offset: float, grid_h: int):
        """[内部辅助] 将 VectorPath 命令转换为 fontTools Pen 命令。"""
        for cmd, *points in path.commands:
            # 坐标转换：从左上角原点(Y向下) -> 字体基线原点(Y向上)
            transformed_points = [(p[0] * scale, (grid_h - p[1]) * scale + y_offset) for p in points]
            
            # 动态调用 pen 对应的方法
            if hasattr(pen, cmd):
                getattr(pen, cmd)(*transformed_points)


# --- 缩略图生成工作器 ---
class ThumbnailWorker(QRunnable):
    """
    一个可在后台线程池中运行的任务，用于生成单个字符的缩略图。
    继承自 QRunnable，是与 QThreadPool 配合使用的标准方式。
    """
    # 定义一个内部信号类，用于在任务完成时发射信号
    class Signals(QObject):
        finished = pyqtSignal(str, QPixmap) # 发射信号时附带 char 和生成的 QPixmap

    def __init__(self, char_obj: 'FontChar', size: int):
        super().__init__()
        self.char_obj = char_obj
        self.size = size
        self.signals = self.Signals()

    def run(self):
        """[后台线程][纯Qt版] 这是在线程池中执行的核心逻辑。"""
        try:
            # 直接调用纯Qt版的预览生成方法
            pixmap = self.char_obj.get_preview_image(all_components={}, size=self.size)
            # 发射完成信号，将结果传递回主线程
            self.signals.finished.emit(self.char_obj.char, pixmap)
        except Exception as e:
            print(f"为字符 '{self.char_obj.char}' 生成缩略图时出错: {e}")


class CharItemDelegate(QStyledItemDelegate):
    """
    自定义委托 (Delegate)，用于完全控制字符列表中的每一个项的绘制。

    此版本实现了高性能的异步缩略图加载机制，并修正了类型注解。
    """
    def __init__(self, list_view: QListView, parent=None):
        """
        初始化委托。

        Args:
            list_view (QListView): 对其服务的 QListView 的引用，用于在任务完成后触发重绘。
            parent (QObject, optional): 父对象。默认为 None。
        """
        super().__init__(parent)
        self.list_view = list_view
        
        # --- 资源预加载 ---
        self.char_font = QFont("Microsoft YaHei UI", 24, QFont.Bold)
        self.info_font = QFont("Microsoft YaHei UI", 8)
        self.item_size = QSize(70, 70)
        self.thumbnail_size = 50 # 缩略图的像素尺寸

        # --- 缓存与线程池 ---
        self.thumbnail_cache: Dict[str, QPixmap] = {}
        self.thread_pool = QThreadPool.globalInstance()
        self.generating: set[str] = set()

    def sizeHint(self, option: 'QStyleOptionViewItem', index: QModelIndex) -> QSize:
        """
        [必要重写][已修正] 向视图提供每个项的推荐大小。

        此方法被 QListView 调用，用于决定为列表中的每一个项目分配多大的空间。
        我们返回一个固定的 QSize，以确保所有字符项在网格视图中大小统一。

        修正: 
        - 将不正确的类型注解 'QStyle.QStyleOptionViewItem' 修正为正确的 'QStyleOptionViewItem'。

        Args:
            option ('QStyleOptionViewItem'): 提供了绘制项的选项和状态信息（在此方法中未使用）。
            index (QModelIndex): 正在查询大小的项的索引（在此方法中未使用）。

        Returns:
            QSize: 每个列表项的固定尺寸。
        """
        # self.item_size 是在 __init__ 方法中定义的 QSize(70, 70)
        return self.item_size

    def paint(self, painter: QPainter, option: 'QStyleOptionViewItem', index: QModelIndex):
        """
        [核心重写][已修正] 自定义列表项的完整绘制逻辑。

        此方法在 QListView 需要绘制或重绘任何一个列表项时被调用。它完全接管了
        默认的绘制行为，实现了高度定制化的外观，包括：
        1.  动态背景和边框颜色，以反映字符的设计状态和用户的交互（选中、悬停）。
        2.  异步加载和缓存已设计字符的缩略图，以避免UI卡顿。
        3.  为未设计的字符渲染清晰的占位符。
        4.  在底部显示字符本身作为标签。

        修正:
        - 将不正确的类型注解 'QStyle.QStyleOptionViewItem' 修正为正确的 'QStyleOptionViewItem'。

        Args:
            painter (QPainter): 用于执行所有绘制操作的画笔工具。
            option ('QStyleOptionViewItem'): 提供了关于当前项的状态（如是否被选中）和
                                            绘制区域（rect）的重要信息。
            index (QModelIndex): 正在绘制的项的模型索引，用于从模型中获取数据。
        """
        # --- 步骤 1: 准备工作 ---
        # 保存 painter 的当前状态，以便在绘制结束后恢复，避免影响其他项的绘制
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing) # 开启抗锯齿以获得平滑的边缘

        # 从模型中获取与此项关联的完整 FontChar 对象
        char_obj: Optional[FontChar] = index.data(Qt.UserRole)
        if not char_obj:
            painter.restore() # 如果没有数据，则不进行任何绘制
            return

        # --- 步骤 2: 计算状态和颜色 ---
        rect = option.rect # 获取此项应被绘制的矩形区域
        is_selected = bool(option.state & QStyle.State_Selected) # 检查项是否被选中
        is_hovered = bool(option.state & QStyle.State_MouseOver) # 检查鼠标是否悬停

        # 定义一套默认颜色
        bg_color = QColor("#f8fafc")
        border_color = QColor("#e2e8f0")
        text_color = QColor("#0f172a")
        muted_text_color = QColor("#94a3b8")

        # 根据字符状态动态调整颜色
        if char_obj.is_designed:
            bg_color, border_color = QColor("#d1fae5"), QColor("#10b981") # 已设计：绿色主题
        elif char_obj.metadata.get('frequency', 0) > 500:
            bg_color, border_color = QColor("#fef3c7"), QColor("#f59e0b") # 高频字：黄色主题

        # 根据交互状态调整颜色
        if is_hovered:
            bg_color = bg_color.lighter(110) # 悬停时背景变亮
        if is_selected:
            border_color = QColor("#6366f1") # 选中时边框变为主题强调色

        # --- 步骤 3: 绘制背景和边框 ---
        painter.setPen(QPen(border_color, 2 if is_selected else 1)) # 选中时边框加粗
        painter.setBrush(QBrush(bg_color))
        # 绘制一个带圆角的矩形作为卡片背景
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 4, 4)

        # --- 步骤 4: 绘制核心内容 (缩略图或占位符) ---
        # 定义预览区域，位于卡片上半部分
        preview_rect = QRect(rect.left(), rect.top(), rect.width(), rect.height() - 16)
        
        if char_obj.is_designed:
            char_key = char_obj.char
            # 4.1 如果缩略图已在缓存中，则直接绘制
            if char_key in self.thumbnail_cache:
                pixmap = self.thumbnail_cache[char_key]
                # 计算 pixmap 的目标矩形，使其在预览区内居中
                target_rect = QRect(0, 0, pixmap.width(), pixmap.height())
                target_rect.moveCenter(preview_rect.center())
                painter.drawPixmap(target_rect.topLeft(), pixmap)
            # 4.2 如果缩略图不在缓存中，则启动后台任务生成它
            else:
                # 在缩略图加载完成前，先绘制一个临时的字符占位符
                painter.setFont(self.char_font)
                painter.setPen(text_color)
                painter.drawText(preview_rect, Qt.AlignCenter, char_obj.char)
                
                # 关键：检查是否已在生成中，防止重复提交任务
                if char_key not in self.generating:
                    self.generating.add(char_key) # 标记为“正在生成”
                    worker = ThumbnailWorker(char_obj, self.thumbnail_size)
                    # 连接信号，当任务完成时会调用 _on_thumbnail_finished
                    worker.signals.finished.connect(self._on_thumbnail_finished)
                    # 将任务提交到全局线程池
                    self.thread_pool.start(worker)
        else:
            # 4.3 对于未设计的字符，直接绘制一个灰色的占位符
            painter.setFont(self.char_font)
            painter.setPen(muted_text_color)
            painter.drawText(preview_rect, Qt.AlignCenter, char_obj.char)

        # --- 步骤 5: 绘制底部信息标签 ---
        painter.setFont(self.info_font)
        painter.setPen(QColor("#475569"))
        # 定义底部信息区域
        info_rect = QRect(rect.left(), rect.bottom() - 16, rect.width(), 14)
        painter.drawText(info_rect, Qt.AlignCenter, char_obj.char)

        # --- 步骤 6: 恢复 painter 状态 ---
        painter.restore()

    def _on_thumbnail_finished(self, char: str, pixmap: QPixmap):
        """
        [主线程槽函数] 当后台缩略图生成任务完成时调用。
        """
        self.thumbnail_cache[char] = pixmap
        if char in self.generating:
            self.generating.remove(char)
        
        model = self.list_view.model()
        if not model: return
        
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            char_obj = model.data(index, Qt.UserRole)
            if char_obj and char_obj.char == char:
                self.list_view.update(index)
                break


# ==============================================================================
# SECTION 3: 自定义UI组件 (CUSTOM WIDGETS)
#
# 这部分定义了应用程序中需要高度自定义外观和行为的核心UI组件。
# 主要的组件是 DrawingCanvas，它是用户进行字体设计的主要工作区。
# ==============================================================================
# --- 主题与样式管理 ---
class LayerItemWidget(QWidget):
    """用于在 QListWidget 中显示的自定义图层项。"""
    # 定义信号，当用户点击可见性或锁定图标时发射
    visibility_changed = pyqtSignal(bool)
    lock_changed = pyqtSignal(bool)

    def __init__(self, stroke_obj: 'HandwritingStroke', parent=None):
        super().__init__(parent)
        self.stroke_obj = stroke_obj

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 1. 可见性按钮
        self.vis_button = QPushButton()
        self.vis_button.setCheckable(True)
        self.vis_button.setChecked(stroke_obj.is_visible)
        self.vis_button.setFixedSize(20, 20)
        self.vis_button.setStyleSheet("QPushButton { border: none; }")
        self._update_vis_icon()
        self.vis_button.clicked.connect(self.on_vis_clicked)
        
        # 2. 图层名称标签
        self.name_label = QLabel(stroke_obj.name)
        
        # 3. 锁定按钮
        self.lock_button = QPushButton()
        self.lock_button.setCheckable(True)
        self.lock_button.setChecked(stroke_obj.is_locked)
        self.lock_button.setFixedSize(20, 20)
        self.lock_button.setStyleSheet("QPushButton { border: none; }")
        self._update_lock_icon()
        self.lock_button.clicked.connect(self.on_lock_clicked)

        layout.addWidget(self.vis_button)
        layout.addWidget(self.name_label, 1) # 占据多余空间
        layout.addWidget(self.lock_button)

    def on_vis_clicked(self, checked):
        self._update_vis_icon()
        self.visibility_changed.emit(checked)

    def on_lock_clicked(self, checked):
        self._update_lock_icon()
        self.lock_changed.emit(checked)
        
    def _update_vis_icon(self):
        icon_path = ":/qt-project.org/styles/commonstyle/images/closed-16.png"
        if self.vis_button.isChecked():
            icon_path = ":/qt-project.org/styles/commonstyle/images/open-16.png" # 简易使用Qt内置图标
        self.vis_button.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton if self.vis_button.isChecked() else QStyle.SP_DialogNoButton))

    def _update_lock_icon(self):
        self.lock_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload if not self.lock_button.isChecked() else QStyle.SP_BrowserStop)) # 简易使用Qt内置图标

class Theme:
    """
    管理应用程序的颜色、字体和全局样式表 (QSS)。
    
    这个类集中了所有的视觉设计参数，使得修改应用外观和实现主题切换变得简单。
    [UI还原版]: 已从Tkinter版本扩充颜色和字体定义，以支持更丰富的视觉层次。
    """
    
    # --- 颜色定义 (已从Tkinter版本扩充) ---
    LIGHT = {
        # 背景
        "bg-primary": "#f8fafc",      # 主背景 (最底层)
        "bg-secondary": "#ffffff",    # 卡片、输入框等内容的背景
        "bg-tertiary": "#f1f5f9",     # 标签页、列表等次级背景
        
        # 内容/文字
        "content-primary": "#0f172a", # 主要文字
        "content-secondary": "#64748b",# 次要文字、提示文字
        
        # 主题强调色
        "accent-primary": "#4f46e5",
        "accent-primary-hover": "#6366f1",
        "accent-primary-pressed": "#4338ca",
        "accent-on-primary": "#ffffff",
        
        # 边框
        "border-primary": "#e2e8f0",
        "border-secondary": "#cbd5e1",
        
        # --- [新增] 语义化颜色 (从Tkinter版还原) ---
        "danger": "#ef4444",   # 危险/删除操作
        "success": "#22c55e",  # 成功/添加操作
        "warning": "#f59e0b",  # 警告操作
        "info": "#3b82f6",     # 信息/中性操作
        "secondary": "#06b6d4" # 次要强调色
    }

    DARK = {
        # 背景
        "bg-primary": "#111827", "bg-secondary": "#1f2937", "bg-tertiary": "#374151",
        # 内容/文字
        "content-primary": "#f9fafb", "content-secondary": "#9ca3af",
        # 主题强调色
        "accent-primary": "#6366f1", "accent-primary-hover": "#818cf8", "accent-primary-pressed": "#4f46e5",
        "accent-on-primary": "#ffffff",
        # 边框
        "border-primary": "#374151", "border-secondary": "#4b5563",
        # 语义化颜色
        "danger": "#f87171", "success": "#4ade80", "warning": "#fbbf24", "info": "#60a5fa", "secondary": "#22d3ee"
    }

    @staticmethod
    def get_font(name: str = "body") -> QFont:
        """
        [已增强] 根据名称获取一个预定义的 QFont 对象。
        处理跨平台字体选择，并提供更丰富的字体层级。
        """
        system = platform.system()
        # 根据操作系统选择合适的默认中文字体
        if system == "Windows":
            default_font_family = "Microsoft YaHei UI"
        elif system == "Darwin":  # macOS
            default_font_family = "PingFang SC"
        else:  # Linux and others
            default_font_family = "Noto Sans CJK SC"

        # --- [新增] 更多字体样式 (从Tkinter版还原) ---
        font_map = {
            "body": QFont(default_font_family, 10),
            "body-bold": QFont(default_font_family, 10, QFont.Bold),
            "h1": QFont(default_font_family, 20, QFont.Bold),
            "h2": QFont(default_font_family, 18, QFont.Bold),
            "h3": QFont(default_font_family, 16, QFont.Bold),
            "h4": QFont(default_font_family, 14, QFont.Bold),
            "h5": QFont(default_font_family, 12, QFont.Bold),
            "title": QFont(default_font_family, 16, QFont.Bold), # 保留别名
            "navbar-title": QFont(default_font_family, 14, QFont.Bold),
            "card-title": QFont(default_font_family, 11, QFont.Bold),
        }
        return font_map.get(name, font_map["body"])

    
    @staticmethod
    def get_qss(theme_name: str = 'light') -> str:
        """
        [最终完整版] 根据主题名称动态生成一个完整的 Qt 样式表 (QSS) 字符串。
        
        此版本包含了所有自定义UI组件和状态的样式规则，包括：
        - 全局字体和颜色设置。
        - 对Dock Widgets, Splitters, ListView, 输入框等标准组件的样式化。
        - 为标签页(QTabWidget)定义了现代化的外观。
        - [新增] 为导航栏大标题(QLabel#NavTitleLabel)定义了专属样式。
        - 为所有按钮定义了通用样式和交互效果。
        - [新增] 为工具按钮(QPushButton[toolButton="true"])定义了独特的样式，
        特别是 :checked 状态下的高亮效果。
        - 为状态栏(QStatusBar)定义了样式。
        """
        colors = Theme.LIGHT if theme_name == 'light' else Theme.DARK

        # 使用 f-string 构建 QSS，易于阅读和修改
        return f"""
            /* 全局设置 */
            QWidget {{
                color: {colors['content-primary']};
                font-family: "{Theme.get_font().family()}";
                font-size: {Theme.get_font().pointSize()}pt;
            }}

            /* 主窗口和可停靠侧边栏 */
            QMainWindow, QDockWidget {{
                background-color: {colors['bg-primary']};
            }}
            
            /* 分割线 */
            QSplitter::handle {{
                background-color: {colors['border-primary']};
            }}
            QSplitter::handle:hover {{
                background-color: {colors['border-secondary']};
            }}

            /* 列表视图 */
            QListView {{
                background-color: {colors['bg-tertiary']};
                border: 1px solid {colors['border-primary']};
                border-radius: 4px;
            }}

            /* 输入框、下拉框、数字选择框 */
            QLineEdit, QComboBox, QSpinBox {{
                background-color: {colors['bg-secondary']};
                border: 1px solid {colors['border-primary']};
                border-radius: 4px;
                padding: 5px 8px;
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
                border: 1px solid {colors['accent-primary']};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            
            /* 标签页 */
            QTabWidget::pane {{
                border: none;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {colors['content-secondary']};
                padding: 8px 20px;
                border-bottom: 2px solid transparent;
                font-weight: bold;
            }}
            QTabBar::tab:hover {{
                color: {colors['content-primary']};
            }}
            QTabBar::tab:selected {{
                color: {colors['accent-primary']};
                border-bottom: 2px solid {colors['accent-primary']};
            }}
            
            /* --- [核心新增] 导航栏大标题专属样式 --- */
            /* 通过对象名选择器 #NavTitleLabel 来精确定位 */
            QLabel#NavTitleLabel {{
                color: white;
                font-family: "{Theme.get_font().family()}";
                font-size: 16pt; /* 直接在这里定义超大字号 */
                font-weight: bold;
                background-color: transparent; /* 确保背景透明，继承ToolBar的背景 */
            }}
            
            /* 按钮 */
            /* 这是所有“主要操作”按钮的通用样式 (例如：保存, 导出) */
            QPushButton {{
                background-color: {colors['accent-primary']};
                color: {colors['accent-on-primary']};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors['accent-primary-hover']};
            }}
            QPushButton:pressed {{
                background-color: {colors['accent-primary-pressed']};
            }}

            /* *** [核心修正] 新增：专门为工具按钮定义的样式 *** */
            /* 使用属性选择器来定位那些被标记为 toolButton 的按钮 */
            QPushButton[toolButton="true"] {{
                background-color: {colors['bg-secondary']};
                color: {colors['content-primary']};
                border: 1px solid {colors['border-primary']};
                font-weight: normal; /* 工具按钮默认使用正常字体粗细 */
                padding: 8px; /* 调整内边距以适应更小的空间 */
                text-align: center; /* 确保文本和图标居中 */
            }}
            QPushButton[toolButton="true"]:hover {{
                background-color: {colors['bg-tertiary']};
                border-color: {colors['border-secondary']};
            }}
            QPushButton[toolButton="true"]:checked {{
                /* 这是实现高亮的关键 */
                background-color: {colors['accent-primary']};
                color: {colors['accent-on-primary']};
                border-color: {colors['accent-primary-pressed']};
                font-weight: bold; /* 选中的工具按钮可以加粗以示突出 */
            }}
            
            /* 状态栏 */
            QStatusBar {{
                background-color: {colors['content-primary']};
                color: {colors['bg-primary']};
            }}
            QStatusBar::item {{
                border: none;
            }}
        """



class CardWidget(QFrame):
    """
    一个自定义的、可重用的、主题感知的卡片式布局容器 (QWidget)。
    
    它提供了一个带标题栏和内容区域的标准视觉结构，用于在 UI 中
    将相关的功能和信息组织在一起，重现了原始 Tkinter 版本中的卡片式设计。

    此版本能够接收一个主题字典，并根据该字典动态设置其颜色，从而支持
    应用程序的全局主题切换。
    
    使用方法:
    1. 获取主题: `current_theme = Theme.LIGHT`
    2. 实例化: `my_card = CardWidget("卡片标题", theme=current_theme)`
    3. 获取内容区并添加组件:
       `layout = QVBoxLayout(my_card.contentWidget())`
       `layout.addWidget(QPushButton("按钮"))`
    4. 切换主题时: `my_card.set_theme(Theme.DARK)`
    """
    def __init__(self, title: str, theme: Dict[str, str], parent: Optional[QWidget] = None):
        """
        [已修正] 初始化卡片组件。

        此版本不再为内容区 (content_widget) 预设布局，从而避免了
        在外部为其添加新布局时产生的 "already has a layout" 警告。
        现在，为 content_widget 设置布局是调用者的责任。

        Args:
            title (str): 显示在卡片顶部的标题文本。
            theme (Dict[str, str]): 包含颜色定义的字典 (例如 Theme.LIGHT)。
            parent (QWidget, optional): 父组件。默认为 None。
        """
        super().__init__(parent)
        self.theme = theme

        # 1. 创建主布局，用于放置标题栏和内容区
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 1) # 细微的边框效果
        main_layout.setSpacing(0) # 标题栏和内容区紧密相连

        # 2. 创建UI组件
        self.title_bar = QLabel(title)
        self.content_widget = QWidget() # 创建内容区容器，但*不*为它设置布局

        # 3. 将组件添加到主布局
        main_layout.addWidget(self.title_bar)
        main_layout.addWidget(self.content_widget, 1) # 内容区占据所有剩余空间

        # 4. 应用初始主题和样式
        self.set_theme(self.theme)

    def set_theme(self, theme: Dict[str, str]):
        """
        [公共接口] 应用一个新的主题字典来更新卡片的外观。
        这是实现全局主题切换的关键方法。
        """
        self.theme = theme
        self.setObjectName("CardWidget")
        self.setStyleSheet(f"""
            #CardWidget {{
                background-color: {self.theme['bg-secondary']};
                border: 1px solid {self.theme['border-primary']};
                border-radius: 6px;
            }}
        """)
        
        self.title_bar.setFont(Theme.get_font("card-title"))
        self.title_bar.setStyleSheet(f"""
            QLabel {{
                background-color: {self.theme['bg-tertiary']};
                color: {self.theme['content-primary']};
                padding: 8px 12px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border-bottom: 1px solid {self.theme['border-primary']};
            }}
        """)

    def setTitle(self, title: str):
        """
        [公共接口] 动态更新卡片的标题。
        
        Args:
            title (str): 新的标题文本。
        """
        self.title_bar.setText(title)

    def setTitleBarColor(self, bg_color: str, text_color: str):
        """
        [公共接口] 覆盖默认的标题栏颜色，用于创建特殊的彩色标题卡片。
        """
        self.title_bar.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                font-weight: bold;
                padding: 8px 12px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border-bottom: none;
            }}
        """)

    def contentWidget(self) -> QWidget:
        """
        [公共接口] 返回内容区的 QWidget，方便外部添加布局和子组件。
        """
        return self.content_widget
class DrawingCanvas(QWidget):
    """
    主绘图画布，是字体设计的核心交互区域。

    此类继承自 QWidget，并重写了其 `paintEvent` 和鼠标事件方法，以实现
    一个完全自定义的绘图表面。它扮演着 MVC 架构中 View 的角色。

    职责:
    - 根据当前选中的 FontChar 对象，负责绘制设计网格、辅助线和所有笔画。
    - 捕获用户的鼠标输入（按下、拖动、释放）。
    - 将原始的鼠标输入转换为设计网格坐标。
    - 在用户绘制时，实时地创建临时的 HandwritingStroke 对象并进行预览。
    - 当一个笔画绘制完成时，通过发射一个 `stroke_finished` 信号来通知
      控制器，将新创建的笔画数据传递出去。
    - 它自身不持有或修改项目的主要数据模型，只负责显示和捕获输入。
    """
    # 定义一个信号，当一个有效的笔画绘制完成时，它会发射这个信号，
    # 并将新创建的 HandwritingStroke 对象作为参数传递出去。
    stroke_finished = pyqtSignal(object)
    stroke_modified = pyqtSignal(int, object, object)
    def __init__(self, main_window: 'MainWindow', parent: Optional[QWidget] = None):
        """
        [最终增强版 V5.5 - 线段与框选支持] 初始化画布。

        此版本新增了支持“线段编辑”和“框选节点”功能所需的状态变量，
        为实现更高级的矢量编辑交互奠定了基础。
        """
        # --- 步骤 1: 调用父类的构造函数 ---
        super().__init__(parent)
        
        # --- 步骤 2: [核心] 保存对主窗口的引用 ---
        # 这是画布与应用程序其他部分通信的关键
        self.main_window = main_window

        # --- 步骤 3: 设置组件的基本行为和外观 ---
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)  # 开启鼠标跟踪以实现悬停效果
        self.setFocusPolicy(Qt.StrongFocus) # 允许画布接收键盘焦点 (用于快捷键)
        self.setCursor(Qt.CrossCursor) # 设置默认光标

        # --- 步骤 4: 初始化核心状态变量 ---
        self.grid_size: int = self.main_window.grid_size
        self.current_char: Optional[FontChar] = None
        self.current_stroke_building: Optional[HandwritingStroke] = None
        self.reference_pixmap: Optional[QPixmap] = None

        # --- 节点编辑状态变量 ---
        self.selected_stroke_index: int = -1
        self.selected_anchor_indices: set[int] = set()
        self.active_anchor_index: int = -1
        self.dragged_node_info: Optional[Dict[str, Any]] = None
        self.pre_modification_stroke: Optional[HandwritingStroke] = None
        self._preview_path: Optional[VectorPath] = None
        self.is_insert_mode: bool = False
        
        # [核心新增] --- 线段编辑状态变量 ---
        self.selected_segment_index: Optional[int] = None # 记录被选中线段的命令索引

        # [核心新增] --- 框选/套索选择状态变量 ---
        self.marquee_rect: Optional[QRectF] = None
        self.marquee_start_pos: Optional[QPoint] = None

        # [核心新增] --- 键盘微调状态变量 ---
        self.is_nudging: bool = False # 标记是否正处于一次连续的键盘微调操作中


        # --- 形状工具状态变量 ---
        self.shape_start_point: Optional[QPointF] = None
        self.shape_current_point: Optional[QPointF] = None
        self.shape_preview_path: Optional[QPainterPath] = None
        
        # --- 显示选项 ---
        self.show_grid: bool = True
        self.show_guides: bool = True
        self.show_metrics: bool = True
        self.show_pro_guides: bool = True

        # --- 内部绘图参数 ---
        self._pixel_size: float = 1.0
        self._grid_rect: QRectF = QRectF()
        self.zhonggong_scale = 0.78
    
    def keyPressEvent(self, event: "QKeyEvent"):
        """[新增] 处理键盘按下事件，主要用于节点的微调 (Nudging)。"""
        # 仅在节点编辑模式且有节点被选中时生效
        if not (self.main_window.current_tool == 'node_editor' and 
                self.selected_stroke_index != -1 and self.selected_anchor_indices):
            super().keyPressEvent(event) # 将事件传递给父类处理
            return
        
        key_map = {
            Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1),
            Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0)
        }
        
        if event.key() in key_map:
            # a. 如果这是本次微调序列的第一次按键
            if not self.is_nudging:
                self.is_nudging = True
                # 记录下操作前的状态，用于最终的撤销命令
                self.pre_modification_stroke = self.current_char.strokes[self.selected_stroke_index].copy()

            # b. 计算位移量
            dx, dy = key_map[event.key()]
            if event.modifiers() & Qt.ShiftModifier:
                dx *= 10; dy *= 10
            
            # c. 直接修改当前笔画对象以实现实时预览
            current_stroke = self.current_char.strokes[self.selected_stroke_index]
            current_path = current_stroke.to_bezier_path()
            
            # (这部分逻辑与 _handle_node_move 类似，但直接作用于当前对象)
            modified_commands = list(current_path.commands)
            for anchor_idx in self.selected_anchor_indices:
                cmd_idx, _, _ = self._find_command_for_anchor(modified_commands, anchor_idx)
                if cmd_idx == -1: continue
                # ... (此处省略了完整的移动命令逻辑，将在一个辅助函数中实现) ...
            
            # 使用一个辅助函数来应用位移，避免代码重复
            new_path = self._apply_nudging_to_path(current_path, self.selected_anchor_indices, dx, dy)
            
            # 重建笔画（但不创建命令），并强制更新预览
            rebuilt_stroke = self._rebuild_stroke_from_path(current_stroke, new_path)
            self.current_char.strokes[self.selected_stroke_index] = rebuilt_stroke
            self.main_window.is_project_dirty = True
            self.update()
            
        else:
            # 如果按下了非方向键，则结束本次微调序列
            self._commit_nudging()
            super().keyPressEvent(event)
    def focusOutEvent(self, event: "QFocusEvent"):
        """[新增] 当画布失去焦点时，确保所有进行中的操作（如微调）被正确提交。"""
        self._commit_nudging()
        super().focusOutEvent(event)

    def _commit_nudging(self):
        """[新增辅助] 将一次连续的键盘微调操作作为一个原子性命令提交到撤销栈。"""
        if self.is_nudging:
            # 创建一个 ModifyStrokeCommand，记录下微调前和最终微调后的状态
            command = ModifyStrokeCommand(
                self.main_window,
                self.selected_stroke_index,
                self.pre_modification_stroke,
                self.current_char.strokes[self.selected_stroke_index]
            )
            command.setText(f"微调 {len(self.selected_anchor_indices)} 个节点")
            self.main_window.undo_stack.push(command)
            
            # 重置微调状态
            self.is_nudging = False
            self.pre_modification_stroke = None
    def _apply_nudging_to_path(self, path: VectorPath, indices_to_move: set, dx: float, dy: float) -> VectorPath:
        """[新增辅助] 将位移应用到路径中指定的节点上，返回新路径。"""
        modified_commands = list(path.commands)
        for anchor_idx in indices_to_move:
            cmd_idx, _, _ = self._find_command_for_anchor(modified_commands, anchor_idx)
            if cmd_idx == -1: continue
            
            cmd = modified_commands[cmd_idx]
            cmd_name = cmd[0]
            
            if cmd_name == 'moveTo':
                p = cmd[1]; modified_commands[cmd_idx] = ('moveTo', (p[0] + dx, p[1] + dy))
            elif cmd_name == 'lineTo':
                p = cmd[1]; modified_commands[cmd_idx] = ('lineTo', (p[0] + dx, p[1] + dy))
            elif cmd_name == 'curveTo':
                _, c1, c2, p1 = cmd
                new_p1 = (p1[0] + dx, p1[1] + dy)
                new_c2 = (c2[0] + dx, c2[1] + dy)
                modified_commands[cmd_idx] = ('curveTo', c1, new_c2, new_p1)
                
                next_cmd_idx = cmd_idx + 1
                if next_cmd_idx < len(modified_commands) and modified_commands[next_cmd_idx][0] == 'curveTo':
                    _, next_c1, next_c2, next_p1 = modified_commands[next_cmd_idx]
                    new_next_c1 = (next_c1[0] + dx, next_c1[1] + dy)
                    modified_commands[next_cmd_idx] = ('curveTo', new_next_c1, next_c2, next_p1)
        
        return VectorPath(modified_commands)

    def contextMenuEvent(self, event: "QContextMenuEvent"):
        """
        [新增][V5.8 上下文菜单] 重写右键菜单事件，提供上下文相关的快捷操作。
        """
        # 仅在节点编辑模式且有笔画被选中时，才显示丰富的上下文菜单
        if not (self.main_window.current_tool == 'node_editor' and self.selected_stroke_index != -1):
            return

        menu = QMenu(self)
        
        # --- 1. 执行命中测试，确定上下文 ---
        hit_node_info = self._hit_test_nodes(event.pos(), self.selected_stroke_index)
        hit_segment_index = self._hit_test_segment(event.pos(), self.selected_stroke_index)

        # --- 2. 根据上下文动态构建菜单 ---

        # 场景 A: 右键点击在节点或其控制柄上
        if hit_node_info:
            # 确保被右键点击的节点也被选中
            if hit_node_info['anchor_index'] not in self.selected_anchor_indices:
                self.selected_anchor_indices = {hit_node_info['anchor_index']}
                self.active_anchor_index = hit_node_info['anchor_index']
                self.update() # 刷新视图以显示新选择

            menu.addAction("转为平滑").triggered.connect(self.main_window.on_convert_to_smooth)
            menu.addAction("转为尖角").triggered.connect(self.main_window.on_convert_to_corner)
            menu.addAction("转为非对称").triggered.connect(self.main_window.on_convert_to_asymmetric)
            menu.addSeparator()
            menu.addAction("删除节点").triggered.connect(self.main_window.on_delete_node)
            
            # 断开路径只对闭合路径上的单个选中节点有效
            stroke = self.current_char.strokes[self.selected_stroke_index]
            if stroke.is_closed and len(self.selected_anchor_indices) == 1:
                menu.addSeparator()
                menu.addAction("在此处断开路径").triggered.connect(self.main_window.on_break_path)

        # 场景 B: 右键点击在线段上 (但不是节点)
        elif hit_segment_index is not None:
            # 确保被右键点击的线段被选中
            self.selected_segment_index = hit_segment_index
            self.selected_anchor_indices.clear() # 清空节点选择
            self.update() # 刷新视图以高亮线段

            menu.addAction("在此处插入节点").triggered.connect(
                lambda: self.main_window.on_insert_node_at_pos(event.pos())
            )
            menu.addSeparator()
            
            # 判断线段类型以决定显示哪个转换选项
            stroke = self.current_char.strokes[self.selected_stroke_index]
            path = stroke.to_bezier_path()
            cmd = path.commands[hit_segment_index]
            if cmd[0] == 'curveTo':
                menu.addAction("转换为直线").triggered.connect(self.main_window.on_convert_segment_to_line)
            elif cmd[0] == 'lineTo':
                 menu.addAction("转换为曲线").triggered.connect(self.main_window.on_convert_segment_to_curve)

        # 场景 C: 右键点击在笔画的空白区域或画布背景
        else:
            # 提供一些通用的选择操作
            action_select_all = menu.addAction("全选所有节点")
            action_select_all.triggered.connect(self.select_all_nodes_in_stroke)
            
            # 如果有选中的节点，则提供取消选择的选项
            if self.selected_anchor_indices:
                action_deselect_all = menu.addAction("取消选择所有节点")
                action_deselect_all.triggered.connect(self.deselect_all_nodes)
            
            # 未来可以添加“粘贴”等操作
            # menu.addSeparator()
            # menu.addAction("粘贴")

        # --- 3. 显示菜单 ---
        # 如果菜单中有任何项目，则在鼠标光标位置显示它
        if menu.actions():
            menu.exec_(event.globalPos())

    def select_all_nodes_in_stroke(self):
        """[新增辅助] 选中当前笔画的所有锚点节点。"""
        if self.selected_stroke_index == -1:
            return
            
        stroke = self.current_char.strokes[self.selected_stroke_index]
        path = stroke.to_bezier_path()
        num_nodes = sum(1 for cmd in path.commands if cmd[0] in ['moveTo', 'lineTo', 'curveTo'])
        
        self.selected_anchor_indices = set(range(num_nodes))
        if self.selected_anchor_indices:
            self.active_anchor_index = next(iter(self.selected_anchor_indices))
        
        self.main_window.update_node_tool_buttons()
        self.update()

    def deselect_all_nodes(self):
        """[新增辅助] 取消对所有节点的选择，但保持笔画的选中状态。"""
        if self.selected_stroke_index == -1:
            return
        
        self.selected_anchor_indices.clear()
        self.active_anchor_index = -1
        
        self.main_window.update_node_tool_buttons()
        self.update()           
    def set_reference_image(self, pixmap: Optional[QPixmap]):
        """[公共接口] 设置或清除参考底模图像。"""
        self.reference_pixmap = pixmap
        self.update()
    def set_char(self, char_obj: Optional[FontChar]):
        """
        [公共接口] 设置当前要在画布上编辑的字符。
        这是控制器更新视图的主要入口。
        
        Args:
            char_obj (Optional[FontChar]): 要显示的字符对象，或None以清空画布。
        """
        self.current_char = char_obj
        self.update() # 请求Qt在下一个事件循环中重绘整个组件

    def set_grid_options(self, show_grid: bool, show_guides: bool, show_metrics: bool, show_pro_guides: bool):
        """[公共接口][已增强] 设置网格、辅助线和度量线的可见性。"""
        self.show_grid = show_grid
        self.show_guides = show_guides
        self.show_metrics = show_metrics
        self.show_pro_guides = show_pro_guides # [核心新增]
        self.update()
    
    # --- 鼠标事件处理 ---
    def mousePressEvent(self, event: "QMouseEvent"):
        """当鼠标在画布上按下时调用，处理所有工具的起始逻辑。"""
        if (event.button() == Qt.LeftButton and 
            self.main_window.current_char_obj and 
            self._grid_rect.contains(event.pos())):
            
            # [核心新增] 如果处于插入节点模式，则处理插入逻辑
            if self.is_insert_mode:
                self.main_window.on_insert_node_at_pos(event.pos())
                return # 处理完插入后直接返回

            tool_id = self.main_window.current_tool
            grid_pos = self._to_grid_coords(event.pos())
            
            if tool_id == 'node_editor':
                modifiers = QApplication.keyboardModifiers()
                self._handle_node_press(event, modifiers)
                return

            # --- [核心新增] 处理形状工具的起始点 ---
            if tool_id in ['line', 'rect', 'circle', 'arc']:
                self.shape_start_point = QPointF(grid_pos[0], grid_pos[1])
                self.shape_current_point = self.shape_start_point
                self.update()
                return

            # --- 保持原有手绘工具的逻辑 ---
            color = self.main_window.current_color.name()
            self.current_stroke_building = HandwritingStroke(stroke_type=tool_id, color=color)
            pressure = 0.8 if self.main_window.pressure_checkbox.isChecked() else 1.0
            self.current_stroke_building.add_point(grid_pos[0], grid_pos[1], pressure)
            self.update()

    def mouseMoveEvent(self, event: "QMouseEvent"):
        """
        [节点处理器][V5.7 最终完整版] 管理鼠标移动事件，完整兼容所有工具和模式。

        此版本整合了所有鼠标移动的逻辑分支，包括手绘、形状工具、节点编辑、
        线段编辑以及框选功能，并确保了它们之间的正确互斥和执行顺序。

        工作流程:
        1.  **拖动逻辑 (鼠标左键按下时)**:
            - **最高优先级：框选**。如果 `marquee_start_pos` 存在，说明正在进行框选，
              则更新选框矩形并立即返回。
            - **次高优先级：节点/线段编辑**。如果当前是节点编辑工具且处于拖动状态
              (`dragged_node_info` 存在)，则调用 `_handle_node_move` 处理节点、句柄
              或线段的拖动，然后返回。
            - **常规优先级：绘图**。如果以上都不是，则执行手绘或形状工具的拖动逻辑。
        2.  **悬停逻辑 (鼠标未按下时)**:
            - 根据当前工具和状态，更新光标以提供交互反馈。
        """
        tool_id = self.main_window.current_tool

        # --- 步骤 1: 处理所有工具的“拖动”逻辑 (当鼠标左键被按下时) ---
        if event.buttons() & Qt.LeftButton:
            
            # a. [最高优先级] 框选拖动
            # 如果 marquee_start_pos 存在，说明用户从空白区域开始拖动，意图是进行框选。
            if self.marquee_start_pos:
                # 根据起始点和当前鼠标位置更新框选矩形
                self.marquee_rect = QRectF(self.marquee_start_pos, event.pos()).normalized()
                self.update() # 请求重绘以在屏幕上实时显示选框
                return # 框选逻辑处理完毕，必须返回

            # b. [次高优先级] 节点、句柄或线段的拖动
            if tool_id == 'node_editor' and self.dragged_node_info:
                self._handle_node_move(event)
                return

            # c. [常规优先级] 绘图/形状工具的拖动
            grid_pos = self._to_grid_coords(event.pos())
            grid_x = max(0, min(self.grid_size, grid_pos[0]))
            grid_y = max(0, min(self.grid_size, grid_pos[1]))
            
            # c.1 形状工具拖动
            if tool_id in ['line', 'rect', 'circle', 'arc'] and self.shape_start_point:
                self.shape_current_point = QPointF(grid_x, grid_y)
                self.update()
                return

            # c.2 手绘工具拖动
            if self.current_stroke_building:
                pressure = 1.0
                if self.main_window.pressure_checkbox.isChecked() and len(self.current_stroke_building.points) > 0:
                    last_point = self.current_stroke_building.points[-1]
                    distance = math.dist((grid_x, grid_y), (last_point[0], last_point[1]))
                    pressure_effect = self.main_window.tool_presets.get(tool_id, {}).get('pressure_effect', 0.5)
                    pressure = max(0.3, min(1.0, 1.0 - distance * 0.05 * pressure_effect))
                self.current_stroke_building.add_point(grid_x, grid_y, pressure)
                self.update()
                return

        # --- 步骤 2: 处理所有工具的“悬停”逻辑 (当鼠标未被按下时) ---
        else:
            # a. 节点编辑模式的悬停
            if tool_id == 'node_editor':
                if self.is_insert_mode:
                    self.setCursor(Qt.CrossCursor)
                    return
                # 在当前选中的笔画上进行节点命中测试以更新光标
                if self.selected_stroke_index != -1:
                    hit_result = self._hit_test_nodes(event.pos(), self.selected_stroke_index)
                    self.setCursor(Qt.OpenHandCursor if hit_result else Qt.ArrowCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            
            # b. 其他工具的悬停
            # 在这种情况下，光标已经在工具切换时（_on_tool_selected）被正确设置，
            # 所以这里不需要执行任何额外的操作。
            else:
                pass
    def _finish_marquee_selection(self, modifiers: Qt.KeyboardModifiers):
        """[新增辅助] 完成框选操作，计算并更新被选中的节点。"""
        if not self.marquee_rect or not self.current_char or self.selected_stroke_index == -1:
            return

        # 1. 将屏幕像素坐标的选框转换为网格坐标
        top_left_grid = self._to_grid_coords(self.marquee_rect.topLeft().toPoint())
        bottom_right_grid = self._to_grid_coords(self.marquee_rect.bottomRight().toPoint())
        grid_marquee = QRectF(QPointF(*top_left_grid), QPointF(*bottom_right_grid)).normalized()

        # 2. 搜集所有在选框内的节点
        nodes_in_rect = set()
        stroke = self.current_char.strokes[self.selected_stroke_index]
        path = stroke.to_bezier_path()
        
        anchor_idx_counter = -1
        for cmd in path.commands:
            point_to_check = None
            is_anchor = False
            if cmd[0] == 'moveTo' or cmd[0] == 'lineTo':
                anchor_idx_counter += 1
                point_to_check = cmd[1]
                is_anchor = True
            elif cmd[0] == 'curveTo':
                anchor_idx_counter += 1
                point_to_check = cmd[3]
                is_anchor = True
            
            if is_anchor and point_to_check and grid_marquee.contains(QPointF(*point_to_check)):
                nodes_in_rect.add(anchor_idx_counter)

        # 3. 根据是否按下 Shift 键，更新最终的选择集合
        is_shift_pressed = bool(modifiers & Qt.ShiftModifier)
        if is_shift_pressed:
            # Shift键：添加到现有选择 (或从中移除，实现异或效果)
            self.selected_anchor_indices = self.selected_anchor_indices.symmetric_difference(nodes_in_rect)
        else:
            # 默认：替换为新选择
            self.selected_anchor_indices = nodes_in_rect
        
        # 4. 如果框选后没有选中任何节点，并且不是在进行 Shift 添加操作，则取消对笔画的选择
        if not self.selected_anchor_indices and not is_shift_pressed:
            self.selected_stroke_index = -1

        # 5. 更新激活锚点
        if self.selected_anchor_indices:
            # 简单地将集合中的第一个元素设为激活锚点
            self.active_anchor_index = next(iter(self.selected_anchor_indices))
        else:
            self.active_anchor_index = -1
    def mouseReleaseEvent(self, event: "QMouseEvent"):
        """
        [已增强][V5.6 框选完成增强版] 完成所有工具的绘制或编辑操作。
        
        此版本新增了完成框选操作的逻辑。
        - 如果 `marquee_rect` 存在，说明用户刚刚完成了一次框选拖拽。
        - 调用辅助函数 `_finish_marquee_selection` 来确定选中的节点。
        - 清理所有框选相关的状态变量。
        - 刷新UI以显示新的选择状态。
        """
        # --- 步骤 1: [核心新增] 完成框选操作 ---
        # 如果 self.marquee_rect 存在，说明用户刚刚完成了一次框选拖拽。
        if self.marquee_rect:
            modifiers = QApplication.keyboardModifiers()
            self._finish_marquee_selection(modifiers)
            
            # 清理框选状态，为下一次操作做准备
            self.marquee_start_pos = None
            self.marquee_rect = None
            
            # 刷新UI以显示新的选择状态
            self.main_window.update_node_tool_buttons()
            self.update()
            
            # 框选操作结束，直接返回
            return

        # --- 步骤 2: 完成节点/线段拖动 ---
        tool_id = self.main_window.current_tool
        if tool_id == 'node_editor':
            # 如果不是框选，并且是节点工具，那么就是常规的节点/线段拖动释放
            self._handle_node_release(event)
            self.setCursor(Qt.ArrowCursor)
            return

        # --- 步骤 3: 完成绘图工具操作 (原有逻辑) ---
        if event.button() == Qt.LeftButton:
            # a. 完成形状工具的绘制
            if tool_id in ['line', 'rect', 'circle', 'arc'] and self.shape_start_point and self.shape_current_point:
                start_p = self.shape_start_point
                end_p = self.shape_current_point
                shape_path = VectorPath()
                
                if tool_id == 'line':
                    shape_path.moveTo(start_p.x(), start_p.y())
                    shape_path.lineTo(end_p.x(), end_p.y())
                elif tool_id == 'rect':
                    shape_path.moveTo(start_p.x(), start_p.y())
                    shape_path.lineTo(end_p.x(), start_p.y())
                    shape_path.lineTo(end_p.x(), end_p.y())
                    shape_path.lineTo(start_p.x(), end_p.y())
                    shape_path.closePath()
                elif tool_id == 'circle':
                    rect = QRectF(start_p, end_p).normalized()
                    kappa = 0.552284749831
                    rx, ry = rect.width() / 2, rect.height() / 2
                    cx, cy = rect.center().x(), rect.center().y()
                    shape_path.moveTo(cx, cy - ry)
                    shape_path.curveTo(cx + kappa * rx, cy - ry, cx + rx, cy - kappa * ry, cx + rx, cy)
                    shape_path.curveTo(cx + rx, cy + kappa * ry, cx + kappa * rx, cy + ry, cx, cy + ry)
                    shape_path.curveTo(cx - kappa * rx, cy + ry, cx - rx, cy + kappa * ry, cx - rx, cy)
                    shape_path.curveTo(cx - rx, cy - kappa * ry, cx - kappa * rx, cy - ry, cx, cy - ry)
                    shape_path.closePath()
                elif tool_id == 'arc':
                    control_p_x = start_p.x() + (end_p.x() - start_p.x()) / 2
                    control_p_y = start_p.y() - abs(end_p.y() - start_p.y())
                    shape_path.moveTo(start_p.x(), start_p.y())
                    shape_path.qCurveTo(control_p_x, control_p_y, end_p.x(), end_p.y())
                
                final_stroke = HandwritingStroke.from_vector_path(shape_path, self.main_window.current_color.name())
                self.stroke_finished.emit(final_stroke)
                
                self.shape_start_point = None
                self.shape_current_point = None
                self.update()
                return

            # b. 完成手绘笔画的绘制
            if self.current_stroke_building:
                final_stroke = self.current_stroke_building
                self.current_stroke_building = None
                if len(final_stroke.points) >= 2:
                    tool_preset = self.main_window.tool_presets.get(tool_id, {})
                    base_smoothing = tool_preset.get('base_smoothing', 0.5)
                    rdp_factor = tool_preset.get('rdp_epsilon_factor', 1.0)
                    final_smoothing = base_smoothing + (self.main_window.stroke_smoothing * 0.4)
                    final_stroke.rdp_epsilon = (2.0 - (final_smoothing * 1.8)) * rdp_factor
                    final_stroke.bezier_tension = 0.3 + (final_smoothing * 0.4)
                    self.stroke_finished.emit(final_stroke)
                self.update()

    # --- 坐标转换辅助函数 ---
    def _to_grid_coords(self, pos: QPoint) -> Tuple[float, float]:
        """将画布的像素坐标转换为抽象的设计网格坐标。"""
        if self._pixel_size == 0: return (0, 0)
        x = (pos.x() - self._grid_rect.x()) / self._pixel_size
        y = (pos.y() - self._grid_rect.y()) / self._pixel_size
        return x, y

    def _to_canvas_coords_path(self, stroke: HandwritingStroke) -> QPainterPath:
        """将一个笔画的网格坐标点列表转换为可被QPainter绘制的QPainterPath。"""
        path = QPainterPath()
        if not stroke or not stroke.points:
            return path
        
        # 将第一个点移至路径起点
        start_point_data = stroke.points[0]
        start_point_qt = QPointF(self._grid_rect.x() + start_point_data[0] * self._pixel_size,
                                 self._grid_rect.y() + start_point_data[1] * self._pixel_size)
        path.moveTo(start_point_qt)
        
        # 连接后续所有点
        for p_data in stroke.points[1:]:
            point_qt = QPointF(self._grid_rect.x() + p_data[0] * self._pixel_size,
                               self._grid_rect.y() + p_data[1] * self._pixel_size)
            path.lineTo(point_qt)
        return path

    # --- 核心绘制方法 ---
    def paintEvent(self, event: "QPaintEvent"):
        """
        [核心重写][V5.7 已修复IndexError] 所有的绘制操作。
        
        此版本修复了在加载新项目或删除笔画后，由于 selected_stroke_index
        变为无效索引而尝试访问列表导致的 `IndexError` 或 `AttributeError`。
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- 步骤 1: 布局计算 ---
        canvas_width, canvas_height = self.width(), self.height()
        padding = 20
        available_size = min(canvas_width - padding * 2, canvas_height - padding * 2)
        self._pixel_size = available_size / self.grid_size if self.grid_size > 0 else 0
        grid_total_size = self._pixel_size * self.grid_size
        start_x = (canvas_width - grid_total_size) / 2
        start_y = (canvas_height - grid_total_size) / 2
        self._grid_rect = QRectF(start_x, start_y, grid_total_size, grid_total_size)

        # --- 步骤 2: 绘制底层元素 ---
        painter.fillRect(self.rect(), QColor(Theme.LIGHT['bg-primary']))
        if self.reference_pixmap:
            painter.drawPixmap(self._grid_rect, self.reference_pixmap, QRectF(self.reference_pixmap.rect()))
        painter.setPen(QPen(QColor(Theme.LIGHT['accent-primary']), 2))
        painter.drawRect(self._grid_rect)

        # --- 步骤 3: 绘制辅助线 ---
        if self.show_grid: self._draw_grid_lines(painter)
        if self.show_guides: self._draw_guide_lines(painter)
        if self.show_metrics: self._draw_metrics_lines(painter)
        if self.show_pro_guides: self._draw_professional_chinese_guides(painter)

        # --- 步骤 4: 绘制已有的笔画和部件 ---
        if self.current_char:
            for i, stroke in enumerate(self.current_char.strokes):
                if not stroke.is_visible: continue
                # 如果是节点编辑模式下被选中的笔画，则跳过，因为它将由顶层的节点绘制函数接管
                if self.main_window.current_tool == 'node_editor' and i == self.selected_stroke_index: continue
                painter.save()
                painter.setOpacity(stroke.opacity)
                self._draw_vector_stroke(painter, stroke)
                painter.restore()
                
            if hasattr(self.main_window, 'components'):
                for instance in self.current_char.components:
                    component = self.main_window.components.get(instance.component_uuid)
                    if not component: continue
                    instance_transform_matrix = instance.get_transform_matrix()
                    for stroke in component.strokes:
                        transformed_stroke = stroke.transform(instance_transform_matrix)
                        self._draw_vector_stroke(painter, transformed_stroke)

        # --- 步骤 5: 绘制正在创建的临时内容 ---
        if self.current_stroke_building: self._draw_raw_stroke(painter, self.current_stroke_building)
        
        if self.main_window.current_tool in ['line', 'rect', 'circle', 'arc'] and self.shape_start_point and self.shape_current_point:
            preview_pen = QPen(QColor(Theme.LIGHT['accent-primary']), 2, Qt.DashLine)
            painter.setPen(preview_pen); painter.setBrush(Qt.NoBrush)
            start_c = QPointF(self._grid_rect.x() + self.shape_start_point.x() * self._pixel_size, self._grid_rect.y() + self.shape_start_point.y() * self._pixel_size)
            current_c = QPointF(self._grid_rect.x() + self.shape_current_point.x() * self._pixel_size, self._grid_rect.y() + self.shape_current_point.y() * self._pixel_size)
            if self.main_window.current_tool == 'line': painter.drawLine(start_c, current_c)
            elif self.main_window.current_tool == 'rect': painter.drawRect(QRectF(start_c, current_c).normalized())
            elif self.main_window.current_tool == 'circle': painter.drawEllipse(QRectF(start_c, current_c).normalized())
            elif self.main_window.current_tool == 'arc':
                path = QPainterPath(start_c)
                control_p_x = start_c.x() + (current_c.x() - start_c.x()) / 2
                control_p_y = start_c.y() - abs(current_c.y() - start_c.y())
                path.quadTo(QPointF(control_p_x, control_p_y), current_c)
                painter.drawPath(path)

        # --- 步骤 6: 绘制节点编辑的顶层UI ---
        if self.main_window.current_tool == 'node_editor':
            
            # [核心修复] 在所有访问 self.current_char.strokes 之前，增加严格的范围检查
            is_stroke_selection_valid = (
                self.current_char and
                0 <= self.selected_stroke_index < len(self.current_char.strokes)
            )

            if self.dragged_node_info and self._preview_path:
                self._draw_nodes_for_path(painter, self._preview_path)
            elif is_stroke_selection_valid:
                stroke = self.current_char.strokes[self.selected_stroke_index]
                if stroke.is_visible:
                    self._draw_nodes_for_path(painter, stroke.to_bezier_path())
            
            if is_stroke_selection_valid and self.selected_segment_index is not None:
                stroke = self.current_char.strokes[self.selected_stroke_index]
                path = stroke.to_bezier_path()
                if self.selected_segment_index < len(path.commands):
                    cmd_to_highlight = path.commands[self.selected_segment_index]
                    segment_path = QPainterPath()
                    p0 = self._get_anchor_point_before(path.commands, self.selected_segment_index)
                    if p0:
                        segment_path.moveTo(QPointF(*p0))
                        if cmd_to_highlight[0] == 'lineTo': segment_path.lineTo(QPointF(*cmd_to_highlight[1]))
                        elif cmd_to_highlight[0] == 'curveTo': segment_path.cubicTo(QPointF(*cmd_to_highlight[1]), QPointF(*cmd_to_highlight[2]), QPointF(*cmd_to_highlight[3]))
                        elif cmd_to_highlight[0] == 'closePath':
                            start_point = self._get_anchor_point_before(path.commands, 1)
                            if start_point: segment_path.lineTo(QPointF(*start_point))
                        
                        transform = QTransform().translate(self._grid_rect.x(), self._grid_rect.y()).scale(self._pixel_size, self._pixel_size)
                        final_segment_path = transform.map(segment_path)
                        
                        highlight_pen = QPen(QColor(Theme.LIGHT['warning']), self.main_window.stroke_width + 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                        highlight_color = highlight_pen.color(); highlight_color.setAlpha(128)
                        highlight_pen.setColor(highlight_color)
                        painter.setPen(highlight_pen); painter.drawPath(final_segment_path)
        
        # --- 步骤 7: 绘制框选矩形 ---
        if self.marquee_rect:
            fill_color = QColor(Theme.LIGHT['accent-primary'])
            fill_color.setAlpha(40)
            painter.setBrush(QBrush(fill_color))
            pen = QPen(QColor(Theme.LIGHT['accent-primary']), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.marquee_rect)
    
    
    
    
    def _rebuild_stroke_from_path(self, stroke: HandwritingStroke, new_path: VectorPath) -> HandwritingStroke:
        """
        [辅助][核心][最终修复版] 从修改后的 VectorPath 重建 HandwritingStroke。
        
        此版本修复了因忘记调用 new_path.is_empty() 方法而导致的 'bool' object is not callable 错误。
        """
        # [核心修正] is_empty 是一个方法，需要使用 () 来调用它
        if new_path.is_empty():
            return stroke.copy()

        # --- 1. 提取新路径的所有锚点坐标 ---
        new_anchor_points_xy = []
        commands = new_path.commands
        if commands and commands[0][0] == 'moveTo':
            new_anchor_points_xy.append(commands[0][1])
            for cmd in commands[1:]:
                if cmd[0] == 'curveTo':
                    new_anchor_points_xy.append(cmd[3])
        
        if not new_anchor_points_xy:
            if commands and commands[0][0] == 'moveTo':
                 new_anchor_points_xy.append(commands[0][1])
            else:
                return stroke.copy()

        # --- 2. [可选但推荐] 插值元数据到新的锚点列表 ---
        new_stroke_points = []
        if stroke.points and len(stroke.points) >= 2:
            original_points = stroke.points
            original_xy = np.array([(p[0], p[1]) for p in original_points])
            original_distances = np.concatenate(([0], np.cumsum(np.linalg.norm(original_xy[1:] - original_xy[:-1], axis=1))))
            original_total_length = original_distances[-1]

            if original_total_length > 1e-6:
                original_norm_dist = original_distances / original_total_length
                original_pressures = np.array([p[2] for p in original_points])
                original_timestamps = np.array([p[3] for p in original_points])
                original_width_factors = np.array([p[4] for p in original_points])
                
                new_xy_array = np.array(new_anchor_points_xy)
                new_distances = np.concatenate(([0], np.cumsum(np.linalg.norm(new_xy_array[1:] - new_xy_array[:-1], axis=1))))
                new_total_length = new_distances[-1]
                new_norm_dist = new_distances / new_total_length if new_total_length > 1e-6 else np.zeros_like(new_distances)

                new_pressures = np.interp(new_norm_dist, original_norm_dist, original_pressures)
                new_timestamps = np.interp(new_norm_dist, original_norm_dist, original_timestamps)
                new_width_factors = np.interp(new_norm_dist, original_norm_dist, original_width_factors)

                for i, (x, y) in enumerate(new_anchor_points_xy):
                    new_stroke_points.append((x, y, new_pressures[i], new_timestamps[i], new_width_factors[i], 0.0))
            else:
                 new_stroke_points = [(p[0], p[1], 1.0, time.time(), 1.0, 0.0) for p in new_anchor_points_xy]
        else:
            new_stroke_points = [(p[0], p[1], 1.0, time.time(), 1.0, 0.0) for p in new_anchor_points_xy]
        
        # --- 3. 创建新笔画对象并注入缓存 ---
        rebuilt_stroke = stroke.copy()
        rebuilt_stroke._raw_points = new_stroke_points # _raw_points 现在只包含锚点
        
        rebuilt_stroke._cached_vector_path = new_path
        
        rebuilt_stroke._last_vectorization_params = (
            rebuilt_stroke.rdp_epsilon, 
            rebuilt_stroke.bezier_tension, 
            rebuilt_stroke.is_closed
        )
        
        return rebuilt_stroke
    
    
    def _draw_nodes_for_path(self, painter: QPainter, vector_path: VectorPath):
        """
        [已重构] 绘制选中笔画的节点和控制手柄的主调度函数。
        """
        if vector_path.is_empty():
            return
        
        # 步骤1: 绘制一层半透明的辉光作为选择高亮
        self._draw_selection_highlight(painter, vector_path)
        
        # 步骤2: 绘制所有节点和活动节点的控制手柄
        self._draw_nodes_and_handles(painter, vector_path)

    def _draw_selection_highlight(self, painter: QPainter, vector_path: VectorPath):
        """[新增辅助] 为选中的路径绘制一个醒目的、半透明的高亮背景。"""
        transform = QTransform().translate(self._grid_rect.x(), self._grid_rect.y()).scale(self._pixel_size, self._pixel_size)
        qpainter_path = vector_path.to_qpainter_path()
        final_path = transform.map(qpainter_path)

        # 绘制一层宽的、半透明的蓝色辉光
        highlight_pen = QPen(QColor(Theme.LIGHT['accent-primary']), self.main_window.stroke_width + 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        highlight_color = highlight_pen.color()
        highlight_color.setAlpha(64) # 设置透明度
        highlight_pen.setColor(highlight_color)
        
        painter.setPen(highlight_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(final_path)
        
        # 在辉光之上再绘制一层细的、不透明的轮廓线
        outline_pen = QPen(QColor(Theme.LIGHT['accent-primary']), 1)
        painter.setPen(outline_pen)
        painter.drawPath(final_path)

    def _draw_nodes_and_handles(self, painter: QPainter, vector_path: VectorPath):
        """[新增辅助] 绘制路径上的所有锚点，以及活动锚点的控制手柄。"""
        node_size, control_size = 4, 3
        anchor_pen = QPen(QColor(Theme.LIGHT['danger']), 1)
        control_pen = QPen(QColor(Theme.LIGHT['info']), 1)
        anchor_brush = QBrush(Qt.white)
        control_brush = QBrush(Qt.white)
        selected_pen = QPen(QColor(Theme.LIGHT['warning']), 2)
        selected_brush = QBrush(QColor(Theme.LIGHT['warning']))
        control_line_pen = QPen(QColor(Theme.LIGHT['content-secondary']), 1, Qt.DashLine)
        transform = QTransform().translate(self._grid_rect.x(), self._grid_rect.y()).scale(self._pixel_size, self._pixel_size)

        # 1. 解析路径，构建一个包含所有节点和句柄信息的字典
        nodes_map = {}
        commands = vector_path.commands
        anchor_idx = 0
        if commands and commands[0][0] == 'moveTo':
            nodes_map[0] = {'pos': commands[0][1], 'c_in': None, 'c_out': None}
            anchor_idx = 1
            for cmd in commands[1:]:
                if cmd[0] == 'curveTo':
                    c1, c2, p1 = cmd[1], cmd[2], cmd[3]
                    # 'c_out' 属于前一个锚点
                    if (anchor_idx - 1) in nodes_map:
                        nodes_map[anchor_idx - 1]['c_out'] = c1
                    # 'c_in' 和 'pos' 属于当前锚点
                    nodes_map[anchor_idx] = {'pos': p1, 'c_in': c2, 'c_out': None}
                    anchor_idx += 1
                elif cmd[0] == 'lineTo':
                    p1 = cmd[1]
                    nodes_map[anchor_idx] = {'pos': p1, 'c_in': None, 'c_out': None}
                    anchor_idx += 1

        # 2. 绘制活动锚点的控制手柄和连接线
        painter.setPen(control_line_pen)
        if self.active_anchor_index in self.selected_anchor_indices and self.active_anchor_index in nodes_map:
            active_node_data = nodes_map[self.active_anchor_index]
            p_canvas = transform.map(QPointF(*active_node_data['pos']))

            if active_node_data['c_in']:
                c_in_canvas = transform.map(QPointF(*active_node_data['c_in']))
                painter.drawLine(p_canvas, c_in_canvas)
                painter.setPen(control_pen); painter.setBrush(control_brush)
                painter.drawEllipse(c_in_canvas, control_size, control_size)

            if active_node_data['c_out']:
                c_out_canvas = transform.map(QPointF(*active_node_data['c_out']))
                painter.drawLine(p_canvas, c_out_canvas)
                painter.setPen(control_pen); painter.setBrush(control_brush)
                painter.drawEllipse(c_out_canvas, control_size, control_size)

        # 3. 绘制所有锚点，并高亮选中的
        for idx, node_data in nodes_map.items():
            is_selected = idx in self.selected_anchor_indices
            p_canvas = transform.map(QPointF(*node_data['pos']))
            painter.setPen(selected_pen if is_selected else anchor_pen)
            painter.setBrush(selected_brush if is_selected else anchor_brush)
            painter.drawRect(QRectF(p_canvas - QPointF(node_size, node_size), QSizeF(node_size * 2, node_size * 2)))
    
    def _hit_test_nodes(self, pos: QPoint, stroke_index: int) -> Optional[Dict[str, Any]]:
        """[私有辅助][最终版] 检查点击是否命中锚点或其句柄，兼容直线与曲线。"""
        if not (self.current_char and 0 <= stroke_index < len(self.current_char.strokes)): 
            return None
        
        stroke = self.current_char.strokes[stroke_index]
        vector_path = stroke.to_bezier_path()
        if vector_path.is_empty(): 
            return None

        hit_tolerance = 8
        transform = QTransform().translate(self._grid_rect.x(), self._grid_rect.y()).scale(self._pixel_size, self._pixel_size)

        # --- 1. [已增强] 收集所有可点击节点的信息，兼容直线段 ---
        nodes_to_test = []
        commands = vector_path.commands
        anchor_idx = 0

        if commands and commands[0][0] == 'moveTo':
            nodes_to_test.append({'type': 'anchor', 'anchor_index': 0, 'pos': commands[0][1]})
            anchor_idx = 1
            
            for seg_idx, cmd in enumerate(commands[1:]):
                if cmd[0] == 'curveTo':
                    c1, c2, p1 = cmd[1], cmd[2], cmd[3]
                    nodes_to_test.append({'type': 'c_out', 'anchor_index': anchor_idx - 1, 'pos': c1, 'seg_idx': seg_idx + 1})
                    nodes_to_test.append({'type': 'c_in', 'anchor_index': anchor_idx, 'pos': c2, 'seg_idx': seg_idx + 1})
                    nodes_to_test.append({'type': 'anchor', 'anchor_index': anchor_idx, 'pos': p1})
                    anchor_idx += 1
                # [核心新增] 处理直线段
                elif cmd[0] == 'lineTo':
                    p1 = cmd[1]
                    nodes_to_test.append({'type': 'anchor', 'anchor_index': anchor_idx, 'pos': p1})
                    anchor_idx += 1
        
        # --- 2. 倒序遍历进行命中测试 (不变) ---
        for node in reversed(nodes_to_test):
            node_pos_c = transform.map(QPointF(*node['pos']))
            if (pos - node_pos_c).manhattanLength() < hit_tolerance:
                return {k: v for k, v in node.items() if k != 'pos'} 
                
        return None

    def _hit_test_segment(self, pos: QPoint, stroke_index: int) -> Optional[int]:
        """
        [新增辅助] 检查点击是否命中笔画的某一个线段。

        Args:
            pos (QPoint): 鼠标点击的画布像素坐标。
            stroke_index (int): 正在测试的笔画在其所属字符中的索引。

        Returns:
            Optional[int]: 如果命中，返回该线段对应的命令在路径命令列表中的索引；
                           否则返回 None。
        """
        if not (self.current_char and 0 <= stroke_index < len(self.current_char.strokes)):
            return None
        
        stroke = self.current_char.strokes[stroke_index]
        vector_path = stroke.to_bezier_path()
        if vector_path.is_empty():
            return None

        transform = QTransform().translate(self._grid_rect.x(), self._grid_rect.y()).scale(self._pixel_size, self._pixel_size)
        
        # 倒序遍历命令，以便优先选中上层的线段（虽然通常不重叠）
        for i in range(len(vector_path.commands) - 1, 0, -1):
            cmd = vector_path.commands[i]
            if cmd[0] not in ['lineTo', 'curveTo', 'closePath']:
                continue

            # 构建一个只包含单个线段的 QPainterPath
            segment_path = QPainterPath()
            p0 = self._get_anchor_point_before(vector_path.commands, i)
            if not p0: continue
            
            segment_path.moveTo(QPointF(*p0))
            if cmd[0] == 'lineTo':
                segment_path.lineTo(QPointF(*cmd[1]))
            elif cmd[0] == 'curveTo':
                segment_path.cubicTo(QPointF(*cmd[1]), QPointF(*cmd[2]), QPointF(*cmd[3]))
            elif cmd[0] == 'closePath':
                start_point = self._get_anchor_point_before(vector_path.commands, 1) # 子路径起点
                if start_point:
                    segment_path.lineTo(QPointF(*start_point))

            # 使用 Stroker 创建热区并进行命中测试
            stroker = QPainterPathStroker()
            stroker.setWidth(10) # 10像素的点击容差
            if stroker.createStroke(transform.map(segment_path)).contains(pos):
                return i # 返回该命令的索引

        return None

    def _get_anchor_point_before(self, commands: list, index: int) -> Optional[Point]:
        """
        [新增辅助] 从命令列表中找到指定索引之前的最后一个锚点坐标。

        这个函数对于处理线段非常关键，因为它能帮助我们确定任何一条
        'lineTo' 或 'curveTo' 线段的起点。

        Args:
            commands (list): 路径命令列表。
            index (int): 当前线段命令在列表中的索引。

        Returns:
            Optional[Point]: 前一个锚点的 (x, y) 坐标，如果找不到则返回 None。
        """
        # 从当前命令的前一个命令开始，向前回溯
        for i in range(index - 1, -1, -1):
            cmd = commands[i]
            # 'moveTo' 和 'lineTo' 的终点就是锚点
            if cmd[0] == 'moveTo' or cmd[0] == 'lineTo':
                return cmd[1]
            # 'curveTo' 的终点也是锚点
            if cmd[0] == 'curveTo':
                return cmd[3]
        # 如果回溯到列表开头都没有找到，说明路径格式有问题或索引无效
        return None  
    
    
    def _handle_node_press(self, event: "QMouseEvent", modifiers: Qt.KeyboardModifiers):
        """
        [节点处理器][V5.5 框选启动增强版] 管理鼠标按下事件。

        此版本新增了对“框选节点”功能的启动逻辑。
        - 当用户在节点编辑模式的空白区域按下鼠标时，不再立即取消所有选择，
          而是初始化一个框选操作，为后续的拖拽框选做准备。
        """
        if not self.current_char or event.button() != Qt.LeftButton:
            return

        # [核心修改] 清空上一次的线段和框选状态
        self.selected_segment_index = None
        self.marquee_start_pos = None # 清除上一次的框选起始点

        # --- 步骤 1: 节点/句柄优先命中测试 ---
        if self.selected_stroke_index != -1:
            hit_result = self._hit_test_nodes(event.pos(), self.selected_stroke_index)
            if hit_result:
                # 命中节点/句柄后，立即处理并返回，不启动框选
                hit_anchor_index = hit_result['anchor_index']
                is_shift_pressed = bool(modifiers & Qt.ShiftModifier)
                if is_shift_pressed:
                    if hit_anchor_index in self.selected_anchor_indices: self.selected_anchor_indices.remove(hit_anchor_index)
                    else: self.selected_anchor_indices.add(hit_anchor_index)
                else:
                    if hit_anchor_index not in self.selected_anchor_indices: self.selected_anchor_indices = {hit_anchor_index}
                self.active_anchor_index = hit_anchor_index
                is_handle = hit_result['type'] in ['c_in', 'c_out']
                can_drag = (hit_anchor_index in self.selected_anchor_indices) or (is_handle and self.active_anchor_index == hit_anchor_index)
                if can_drag:
                    self.pre_modification_stroke = self.current_char.strokes[self.selected_stroke_index].copy()
                    drag_info = hit_result.copy()
                    drag_info['start_mouse_pos'] = self._to_grid_coords(event.pos())
                    if hit_result['type'] == 'anchor':
                        drag_info['original_positions'] = {}
                        base_path = self.pre_modification_stroke.to_bezier_path()
                        for idx in self.selected_anchor_indices:
                            pos = self._get_node_pos_from_path(base_path, {'type': 'anchor', 'anchor_index': idx})
                            if pos: drag_info['original_positions'][idx] = pos
                    self.dragged_node_info = drag_info
                    self.setCursor(Qt.ClosedHandCursor)
                self.main_window.update_node_tool_buttons()
                self.update()
                return

        # --- 步骤 2: 笔画和线段命中测试 ---
        clicked_stroke_index = -1
        for i, stroke in reversed(list(enumerate(self.current_char.strokes))):
            if not stroke.is_visible or stroke.is_locked: continue
            path = stroke.to_bezier_path().to_qpainter_path()
            transform = QTransform().translate(self._grid_rect.x(), self._grid_rect.y()).scale(self._pixel_size, self._pixel_size)
            stroker = QPainterPathStroker(); stroker.setWidth(10)
            if stroker.createStroke(transform.map(path)).contains(event.pos()):
                clicked_stroke_index = i; break

        if clicked_stroke_index != -1:
            # 命中笔画，继续进行线段命中测试
            hit_segment_index = self._hit_test_segment(event.pos(), clicked_stroke_index)
            
            if hit_segment_index is not None:
                # 精确命中线段
                self.selected_stroke_index = clicked_stroke_index
                self.selected_segment_index = hit_segment_index
                self.selected_anchor_indices.clear()
                self.active_anchor_index = -1
                self.pre_modification_stroke = self.current_char.strokes[self.selected_stroke_index].copy()
                self.dragged_node_info = {
                    'type': 'segment',
                    'segment_index': hit_segment_index,
                    'start_mouse_pos': self._to_grid_coords(event.pos())
                }
                self.setCursor(Qt.ClosedHandCursor)
            else:
                # 只命中了笔画的“空白”部分
                if self.selected_stroke_index != clicked_stroke_index:
                    self.selected_stroke_index = clicked_stroke_index
                    self.selected_anchor_indices.clear()
                    self.active_anchor_index = -1
                    self.dragged_node_info = None
        else:
            # --- [核心新增] 步骤 3: 启动框选 ---
            # 如果点击了空白区域，则不立即取消选择，而是记录框选的起始点
            self.marquee_start_pos = event.pos()
            # 只有在鼠标释放时如果没有形成有效的框选矩形，我们才真正取消选择。
            # 为了交互的即时性，可以先清空节点选择，但保留笔画选择
            if not (modifiers & Qt.ShiftModifier):
                 self.selected_anchor_indices.clear()
                 self.active_anchor_index = -1

        # --- 步骤 4: 统一刷新 ---
        self.main_window.update_node_tool_buttons()
        self.update()


    def _handle_node_move(self, event: "QMouseEvent"):
        """
        [节点处理器][V5.6 框选拖拽增强版] 管理鼠标在节点编辑模式下的拖动事件。

        此版本新增了对“框选节点”功能的拖动逻辑支持。

        工作流程:
        1.  **模式判断**: 首先检查是否处于框选模式（通过 `self.marquee_start_pos` 是否存在）。
        2.  **框选逻辑**:
            - 如果是框选模式，则根据鼠标当前位置和起始点计算并更新
              `self.marquee_rect` (框选矩形)。
            - 调用 `self.update()` 实时绘制选框，并立即返回。
        3.  **原有拖动逻辑**: 如果不是框选模式，则执行之前已有的、针对节点、
            句柄和线段的拖动变形逻辑。
        """
        # --- 步骤 1: [核心新增] 处理框选拖动 ---
        # 如果 self.marquee_start_pos 存在，说明用户从空白区域开始拖动，
        # 意图是进行框选。
        if self.marquee_start_pos:
            # 根据起始点和当前鼠标位置创建一个 QRectF 对象
            self.marquee_rect = QRectF(self.marquee_start_pos, event.pos()).normalized()
            # 请求重绘以在屏幕上显示选框
            self.update()
            # 框选逻辑处理完毕，直接返回，不执行后续的节点拖动逻辑
            return

        # --- 步骤 2: 处理节点、句柄、线段的拖动 (原有逻辑) ---
        if not self.dragged_node_info or not self.pre_modification_stroke:
            return

        current_mouse_pos_grid = self._to_grid_coords(event.pos())
        start_mouse_pos_grid = self.dragged_node_info['start_mouse_pos']
        offset_x = current_mouse_pos_grid[0] - start_mouse_pos_grid[0]
        offset_y = current_mouse_pos_grid[1] - start_mouse_pos_grid[1]

        base_path = self.pre_modification_stroke.to_bezier_path()
        modified_commands = list(base_path.commands)
        dragged_node_type = self.dragged_node_info['type']
        
        # 场景 C: 正在拖动一个线段
        if dragged_node_type == 'segment':
            seg_idx = self.dragged_node_info['segment_index']
            cmd_to_modify = modified_commands[seg_idx]
            
            if cmd_to_modify[0] == 'lineTo':
                p0 = np.array(self._get_anchor_point_before(modified_commands, seg_idx))
                p1 = np.array(cmd_to_modify[1])
                mid_point = (p0 + p1) / 2.0
                vec = p1 - p0
                normal = np.array([-vec[1], vec[0]])
                norm_mag = np.linalg.norm(normal)
                if norm_mag > 1e-6:
                    normal /= norm_mag
                mouse_vec = np.array([offset_x, offset_y])
                proj_len = np.dot(mouse_vec, normal)
                control_point = mid_point + normal * proj_len * 2
                c1 = tuple(p0 * 0.25 + control_point * 0.75)
                c2 = tuple(p1 * 0.25 + control_point * 0.75)
                modified_commands[seg_idx] = ('curveTo', c1, c2, tuple(p1))
            elif cmd_to_modify[0] == 'curveTo':
                _, c1_orig, c2_orig, p1 = cmd_to_modify
                new_c1 = (c1_orig[0] + offset_x, c1_orig[1] + offset_y)
                new_c2 = (c2_orig[0] + offset_x, c2_orig[1] + offset_y)
                modified_commands[seg_idx] = ('curveTo', new_c1, new_c2, p1)
        
        # 其他场景: 正在拖动节点或句柄
        else:
            is_alt_pressed = bool(QApplication.keyboardModifiers() & Qt.AltModifier)
            if dragged_node_type == 'anchor':
                dragged_anchor_idx = self.dragged_node_info['anchor_index']
                p_orig = self._get_node_pos_from_path(base_path, {'type': 'anchor', 'anchor_index': dragged_anchor_idx})
                if not p_orig: return
                new_p = (p_orig[0] + offset_x, p_orig[1] + offset_y)
                cmd_idx, _, _ = self._find_command_for_anchor(base_path.commands, dragged_anchor_idx)
                if cmd_idx == -1: return
                cmd_name = base_path.commands[cmd_idx][0]
                if cmd_name == 'moveTo': modified_commands[cmd_idx] = ('moveTo', new_p)
                elif cmd_name == 'lineTo': modified_commands[cmd_idx] = ('lineTo', new_p)
                elif cmd_name == 'curveTo':
                    _, c1_orig, c2_orig, _ = base_path.commands[cmd_idx]
                    new_c2 = (c2_orig[0] + offset_x, c2_orig[1] + offset_y)
                    modified_commands[cmd_idx] = ('curveTo', c1_orig, new_c2, new_p)
                    next_cmd_idx = cmd_idx + 1
                    if next_cmd_idx < len(base_path.commands) and base_path.commands[next_cmd_idx][0] == 'curveTo':
                        _, next_c1_orig, next_c2_orig, next_p1_orig = base_path.commands[next_cmd_idx]
                        new_next_c1 = (next_c1_orig[0] + offset_x, next_c1_orig[1] + offset_y)
                        modified_commands[next_cmd_idx] = ('curveTo', new_next_c1, next_c2_orig, next_p1_orig)
            else: # c_in or c_out
                anchor_idx = self.dragged_node_info['anchor_index']
                stroke = self.pre_modification_stroke
                anchor_type = stroke.anchor_types[anchor_idx] if 0 <= anchor_idx < len(stroke.anchor_types) else 'corner'
                handle_orig_pos = self._get_node_pos_from_path(base_path, self.dragged_node_info)
                if not handle_orig_pos: return
                new_handle_pos = np.array([handle_orig_pos[0] + offset_x, handle_orig_pos[1] + offset_y])
                self._update_control_point_in_commands(modified_commands, self.dragged_node_info, tuple(new_handle_pos))
                if anchor_type == 'smooth' and not is_alt_pressed:
                    anchor_pos = np.array(self._get_node_pos_from_path(base_path, {'type': 'anchor', 'anchor_index': anchor_idx}))
                    mirror_handle_info = self.dragged_node_info.copy()
                    mirror_handle_info['type'] = 'c_in' if dragged_node_type == 'c_out' else 'c_out'
                    new_mirror_pos = 2 * anchor_pos - new_handle_pos
                    self._update_control_point_in_commands(modified_commands, mirror_handle_info, tuple(new_mirror_pos))

        self._preview_path = VectorPath(modified_commands)
        self.update()
    def _update_control_point_in_commands(self, commands: list, handle_info: dict, new_pos: tuple):
        """[新增辅助] 在命令列表中精确更新一个控制点的位置。"""
        seg_idx = handle_info.get('seg_idx')
        if seg_idx is not None and 0 <= seg_idx < len(commands):
            cmd = commands[seg_idx]
            if cmd[0] == 'curveTo':
                c1, c2, p1 = cmd[1], cmd[2], cmd[3]
                if handle_info['type'] == 'c_out':
                    commands[seg_idx] = ('curveTo', new_pos, c2, p1)
                elif handle_info['type'] == 'c_in':
                    commands[seg_idx] = ('curveTo', c1, new_pos, p1)
    
    
    def _find_command_for_anchor(self, commands: List[PathCommand], target_anchor_idx: int) -> Tuple[int, Optional[PathCommand], int]:
        """
        [新增辅助][已修复] 查找包含指定索引锚点的命令、其在列表中的索引以及其锚点索引。

        此函数是节点编辑逻辑的核心辅助工具。它会遍历命令列表，并像解析器一样
        对锚点进行计数。当计数器与目标锚点索引匹配时，它会返回所有需要的信息。

        Args:
            commands (List[PathCommand]): 要搜索的路径命令列表。
            target_anchor_idx (int): 目标锚点的索引 (从0开始)。

        Returns:
            Tuple[int, Optional[PathCommand], int]: 一个包含三个元素的元组：
                - 命令在列表中的索引 (如果未找到，则为 -1)。
                - 命令本身 (如果未找到，则为 None)。
                - 找到的锚点的索引 (如果未找到，则为 -1)。
        """
        anchor_counter = -1
        for i, cmd in enumerate(commands):
            # 只有 moveTo, lineTo, 和 curveTo 命令的终点才被视为锚点
            if cmd[0] in ['moveTo', 'lineTo', 'curveTo']:
                anchor_counter += 1
                if anchor_counter == target_anchor_idx:
                    return i, cmd, anchor_counter
        return -1, None, -1
    def _handle_node_release(self, event: "QMouseEvent"):
        """
        [节点处理器][V5.4 最终增强版 - Alt键类型转换] 管理鼠标释放事件。

        此版本新增了处理快捷键交互的核心逻辑：如果在拖动一个“平滑”节点的
        控制柄时按下了 Alt 键，那么在操作结束后，该节点的类型将被自动转换为
        “非对称”，并将此类型变化与几何变化一同记录到撤销历史中。

        工作流程:
        1.  **前置检查**: 确认是否存在有效的拖动操作。
        2.  **数据重建**: 从 `_preview_path` 重建基础的 `rebuilt_stroke`。
        3.  **[核心新增] 类型转换检查**:
            - 检查拖动期间 `Alt` 键是否被按下。
            - 检查被拖动的节点是否是控制柄 (`c_in`/`c_out`)。
            - 检查该控制柄所属的锚点在*拖动前*的类型是否为 `smooth`。
            - 如果所有条件满足，则在 `rebuilt_stroke` 中将该锚点的类型更新为 `asymmetric`。
        4.  **提交撤销命令**: 将包含了所有几何和属性变化的 `rebuilt_stroke` 与
            拖动前的 `pre_modification_stroke` 一同封装进 `ModifyStrokeCommand`。
        5.  **状态清理**: 彻底清理所有与本次拖动相关的临时状态变量。
        """
        # --- 步骤 0: 检查是否存在一次有效的拖动操作 ---
        if not self.dragged_node_info or not self.pre_modification_stroke:
            self.dragged_node_info = None
            self.pre_modification_stroke = None
            self._preview_path = None
            self.setCursor(Qt.ArrowCursor)
            return

        rebuilt_stroke = None
        modification_occurred = False

        # --- 步骤 1: 从预览路径重建最终的笔画对象 ---
        if self._preview_path and not self._preview_path.is_empty():
            rebuilt_stroke = self._rebuild_stroke_from_path(
                self.pre_modification_stroke,
                self._preview_path
            )

            # --- [核心新增] 处理因 Alt 键拖动导致的节点类型变化 ---
            modifiers = QApplication.keyboardModifiers()
            is_alt_pressed = bool(modifiers & Qt.AltModifier)
            dragged_node_type = self.dragged_node_info['type']
            
            # 仅当用 Alt 拖动控制柄时才触发类型转换
            if is_alt_pressed and rebuilt_stroke and dragged_node_type in ['c_in', 'c_out']:
                anchor_idx = self.dragged_node_info['anchor_index']
                original_stroke = self.pre_modification_stroke # 必须从拖动前的状态获取类型
                
                # 检查原始类型是否为 'smooth'
                if 0 <= anchor_idx < len(original_stroke.anchor_types) and \
                   original_stroke.anchor_types[anchor_idx] == 'smooth':
                    
                    # 在重建后的笔画中，将该节点的类型更新为 'asymmetric'
                    rebuilt_stroke.anchor_types[anchor_idx] = 'asymmetric'

            # 检查最终的 rebuilt_stroke 是否与原始笔画有实质性差异
            if rebuilt_stroke and rebuilt_stroke.to_dict() != self.pre_modification_stroke.to_dict():
                modification_occurred = True

        # --- 步骤 2: 创建并提交撤销命令 ---
        if modification_occurred and rebuilt_stroke:
            command = ModifyStrokeCommand(
                main_window=self.main_window,
                stroke_index=self.selected_stroke_index,
                old_stroke=self.pre_modification_stroke,
                new_stroke=rebuilt_stroke
            )
            self.main_window.undo_stack.push(command)
        else:
            self.main_window._update_all_views()

        # --- 步骤 3: 清理所有与本次拖动相关的临时状态 (关键步骤) ---
        self.dragged_node_info = None
        self.pre_modification_stroke = None
        self._preview_path = None

        # --- 步骤 4: 恢复默认的箭头光标 ---
        self.setCursor(Qt.ArrowCursor)
    
    # --- [新增] 高级节点编辑方法 ---

    def close_selected_path(self):
        """将当前选中的开放路径闭合。"""
        if not self.current_char or self.selected_stroke_index == -1: return
        
        old_stroke = self.current_char.strokes[self.selected_stroke_index]
        path = old_stroke.to_bezier_path()
        if not path.commands or path.commands[-1][0] == 'closePath': return

        new_commands = list(path.commands)
        new_commands.append(('closePath',))
        new_path = VectorPath(new_commands)
        
        new_stroke = self._rebuild_stroke_from_path(old_stroke, new_path)
        new_stroke.is_closed = True

        cmd = ModifyStrokeCommand(self.main_window, self.selected_stroke_index, old_stroke, new_stroke)
        self.main_window.undo_stack.push(cmd)

    def break_path_at_selected_node(self):
        """[最终修正版 V5 - 绝对保真] 在选中的节点处精确地“剪开”路径，不改变形状。"""
        if not (self.current_char and 
                self.selected_stroke_index != -1 and
                len(self.selected_anchor_indices) == 1):
            return

        old_stroke = self.current_char.strokes[self.selected_stroke_index].copy()
        path = old_stroke.to_bezier_path()
        
        if not path.commands or path.commands[-1][0] != 'closePath':
            return

        node_idx_to_break = next(iter(self.selected_anchor_indices))
        
        # --- 1. 获取路径上所有锚点的坐标 ---
        all_anchor_points = []
        if path.commands and path.commands[0][0] == 'moveTo':
            all_anchor_points.append(path.commands[0][1])
            for cmd in path.commands[1:-1]: # 忽略最后的 closePath
                if cmd[0] == 'lineTo':
                    all_anchor_points.append(cmd[1])
                elif cmd[0] == 'curveTo':
                    all_anchor_points.append(cmd[3])
        
        num_points = len(all_anchor_points)
        if num_points < 2 or node_idx_to_break >= num_points:
            return

        # --- 2. [核心逻辑] 旋转坐标列表，并将起点复制到结尾 ---
        # a. 旋转列表，使断开点成为新的起点
        new_point_sequence = all_anchor_points[node_idx_to_break:] + all_anchor_points[:node_idx_to_break]
        
        # b. [关键] 将新的起点（即断开点）复制一份并添加到列表末尾，形成新的终点
        new_point_sequence.append(new_point_sequence[0])
        
        # --- 3. [核心逻辑] 从新的点列表重建开放路径 ---
        # a. 创建一个临时的 HandwritingStroke 对象
        temp_stroke = HandwritingStroke(
            points=[(p[0], p[1], 1.0, time.time(), 1.0, 0.0) for p in new_point_sequence],
            is_closed=False # 明确指定为开放路径
        )
        
        # b. 让这个临时笔画自己生成一个最优的、开放的三次贝塞尔路径
        new_path = temp_stroke.to_bezier_path()

        # --- 4. 创建最终的新笔画对象并执行命令 ---
        new_stroke = self._rebuild_stroke_from_path(old_stroke, new_path)
        new_stroke.is_closed = False
        # 确保新笔画的原始点与我们创建的序列一致
        new_stroke._raw_points = temp_stroke._raw_points

        cmd = ModifyStrokeCommand(self.main_window, self.selected_stroke_index, old_stroke, new_stroke)
        self.main_window.undo_stack.push(cmd)

        # 断开后，节点选择状态需要重置
        self.selected_anchor_indices.clear()
        self.active_anchor_index = -1

    def delete_selected_node(self):
        """删除选中的节点。"""
        # [核心修正] 使用 self.selected_anchor_indices
        if not self.current_char or self.selected_stroke_index == -1 or not self.selected_anchor_indices: return
        
        old_stroke = self.current_char.strokes[self.selected_stroke_index]
        path = old_stroke.to_bezier_path()
        
        # [核心修正] 从新的集合获取索引
        indices_to_delete = self.selected_anchor_indices
        
        new_commands = []
        anchor_idx_counter = -1
        for cmd in path.commands:
            is_anchor = cmd[0] in ['moveTo', 'lineTo', 'curveTo']
            if is_anchor:
                anchor_idx_counter += 1
            
            if not (is_anchor and anchor_idx_counter in indices_to_delete):
                new_commands.append(cmd)
        
        if new_commands and new_commands[0][0] != 'moveTo':
            next_cmd = new_commands[0]
            start_point = None
            if next_cmd[0] == 'lineTo': start_point = next_cmd[1]
            elif next_cmd[0] == 'curveTo': start_point = next_cmd[3]
            if start_point:
                new_commands[0] = ('moveTo', start_point)
        
        new_path = VectorPath(new_commands)
        new_stroke = self._rebuild_stroke_from_path(old_stroke, new_path)
        
        cmd = ModifyStrokeCommand(self.main_window, self.selected_stroke_index, old_stroke, new_stroke)
        self.main_window.undo_stack.push(cmd)
        # 清空选择，因为索引会改变
        self.selected_anchor_indices.clear()

    def convert_segment_to_line(self):
        """将选中节点之后的那一段路径转换为直线。"""
        # [核心修正] 使用 self.selected_anchor_indices
        if not self.current_char or self.selected_stroke_index == -1 or not self.selected_anchor_indices: return
        if len(self.selected_anchor_indices) != 1: return

        node_idx = next(iter(self.selected_anchor_indices))
        old_stroke = self.current_char.strokes[self.selected_stroke_index]
        path = old_stroke.to_bezier_path()
        commands = list(path.commands)
        
        cmd_idx_of_node = node_idx
        if cmd_idx_of_node >= len(commands) or commands[cmd_idx_of_node][0] != 'curveTo':
            return 

        next_anchor_pos = commands[cmd_idx_of_node][3]
        commands[cmd_idx_of_node] = ('lineTo', next_anchor_pos)
        
        new_path = VectorPath(commands)
        new_stroke = self._rebuild_stroke_from_path(old_stroke, new_path)
        
        cmd = ModifyStrokeCommand(self.main_window, self.selected_stroke_index, old_stroke, new_stroke)
        self.main_window.undo_stack.push(cmd)

    def convert_segment_to_curve(self):
        """将选中节点之后的那一段直线转换为曲线。"""
        # [核心修正] 使用 self.selected_anchor_indices
        if not self.current_char or self.selected_stroke_index == -1 or not self.selected_anchor_indices: return
        if len(self.selected_anchor_indices) != 1: return

        node_idx = next(iter(self.selected_anchor_indices))
        old_stroke = self.current_char.strokes[self.selected_stroke_index]
        path = old_stroke.to_bezier_path()
        commands = list(path.commands)

        cmd_idx_of_node = node_idx
        if cmd_idx_of_node >= len(commands) or commands[cmd_idx_of_node][0] != 'lineTo':
            return

        p0 = self._get_node_pos_from_path(path, {'type': 'anchor', 'anchor_index': node_idx - 1})
        p1 = commands[cmd_idx_of_node][1]
        if not p0: return

        c1 = (p0[0] * 2/3 + p1[0] * 1/3, p0[1] * 2/3 + p1[1] * 1/3)
        c2 = (p0[0] * 1/3 + p1[0] * 2/3, p0[1] * 1/3 + p1[1] * 2/3)

        commands[cmd_idx_of_node] = ('curveTo', c1, c2, p1)
        
        new_path = VectorPath(commands)
        new_stroke = self._rebuild_stroke_from_path(old_stroke, new_path)

        cmd = ModifyStrokeCommand(self.main_window, self.selected_stroke_index, old_stroke, new_stroke)
        self.main_window.undo_stack.push(cmd)
    
    def _draw_metrics_lines(self, painter: QPainter):
        """
        [私有辅助][UI还原最终数学修正版] 绘制精确定位的专业度量线。
        
        此版本修正了坐标计算的根本逻辑，确保所有度量线都根据其
        相对于 UPM (grid_size) 的标准比例，被精确地绘制在网格内的
        正确位置上，完美解决了“拉伸”问题。
        """
        metrics_ratios = {
            "上伸部": 0.88, "大写高度": 0.70, "x 高度": 0.48,
            "基线": 0.0, "下伸部": -0.12,
        }
        color_map = {
            "上伸部": "info", "大写高度": "warning", "x 高度": "warning",
            "基线": "success", "下伸部": "info",
        }

        # --- [核心修正] 使用最直接的坐标计算逻辑 ---

        # 1. 获取网格的几何基准
        grid_top = self._grid_rect.top()
        grid_height = self._grid_rect.height()
        
        # 2. 循环绘制每一条线
        for name, ratio in metrics_ratios.items():
            # a. 计算当前度量线在画布上的绝对 Y 坐标
            #    公式: Y = 网格顶部 + (上伸部比例 - 当前比例) * UPM对应的像素高度
            #    UPM 对应的像素高度就是 grid_height
            #    上伸部比例是 0.88，这是我们的 Y=0 参考点（相对于字体坐标系）
            y = grid_top + (0.88 - ratio) * grid_height

            # --- 后续绘制逻辑保持不变 ---
            color_key = color_map.get(name, "content-secondary")
            line_color = QColor(Theme.LIGHT[color_key])
            line_color.setAlpha(200)

            pen = QPen(line_color, 1, Qt.DashDotLine)
            painter.setPen(pen)
            
            painter.drawLine(QPointF(self._grid_rect.left() - 20, y), 
                             QPointF(self._grid_rect.right() + 20, y))
            
            painter.setPen(QColor(Theme.LIGHT[color_key])) 
            painter.setFont(Theme.get_font("body"))
            text_rect = QRectF(self._grid_rect.left() - 120, y - 15, 100, 30)
            painter.drawText(text_rect, Qt.AlignRight | Qt.AlignVCenter, name)
    def _draw_grid_lines(self, painter: QPainter):
        """
        [私有辅助][UI还原最终双层网格版] 绘制具有清晰层级的专业双层网格。
        
        此版本精确实现了“先绘制8x8主网格，再在每个主网格内绘制8x8次网格”
        的专业视觉效果，完全复现了参考图的层次感。
        """
        # --- 1. 定义视觉样式 ---
        # 主网格线: 中性深灰色，实线，较粗
        major_pen = QPen(QColor(Theme.LIGHT['content-secondary']), 1.5, Qt.SolidLine)
        
        # 次网格线: 非常浅的灰色，实线，最细
        minor_pen = QPen(QColor(Theme.LIGHT['border-primary']), 1.0, Qt.SolidLine)

        # --- 2. 绘制 8x8 的主网格 ---
        painter.setPen(major_pen)
        # 计算主网格每一格的像素尺寸
        major_cell_size = self._grid_rect.width() / 8.0
        
        for i in range(1, 8):
            # 绘制垂直主线
            x = self._grid_rect.left() + i * major_cell_size
            painter.drawLine(QPointF(x, self._grid_rect.top()), QPointF(x, self._grid_rect.bottom()))
            
            # 绘制水平主线
            y = self._grid_rect.top() + i * major_cell_size
            painter.drawLine(QPointF(self._grid_rect.left(), y), QPointF(self._grid_rect.right(), y))

        # --- 3. 在每个主网格内绘制 8x8 的次网格 ---
        painter.setPen(minor_pen)
        # 计算次网格每一格的像素尺寸
        minor_cell_size = major_cell_size / 8.0
        
        # 遍历所有 64 个主网格单元格
        for row in range(8):
            for col in range(8):
                # 计算当前主单元格的左上角坐标
                cell_start_x = self._grid_rect.left() + col * major_cell_size
                cell_start_y = self._grid_rect.top() + row * major_cell_size

                # 在这个主单元格内部绘制 7 条垂直和 7 条水平的次网格线
                for i in range(1, 8):
                    # 绘制垂直次线
                    x = cell_start_x + i * minor_cell_size
                    painter.drawLine(QPointF(x, cell_start_y), QPointF(x, cell_start_y + major_cell_size))

                    # 绘制水平次线
                    y = cell_start_y + i * minor_cell_size
                    painter.drawLine(QPointF(cell_start_x, y), QPointF(cell_start_x + major_cell_size, y))
    def _draw_guide_lines(self, painter: QPainter):
        """
        [私有辅助][UI还原最终版 + 中文辅助线] 绘制九宫格、米字格及天地中线辅助线。
        此版本在原有米字格基础上，精确添加了中文字符设计所需的天、地、中线
        以及框线标签，完美复现了参考UI的专业辅助线系统。
        """
        # --- 1. 设置画笔和字体样式 ---
        guide_color = QColor(Theme.LIGHT['info'])
        guide_color.setAlpha(128)
        pen = QPen(guide_color, 1, Qt.DashLine)
        painter.setPen(pen)
        
        # 用于绘制标签的字体
        label_font = Theme.get_font("body")
        label_font.setPointSize(8) # 使用稍小的字号
        painter.setFont(label_font)
        
        # --- 2. 获取网格的几何信息 ---
        left, top = self._grid_rect.left(), self._grid_rect.top()
        width, height = self._grid_rect.width(), self._grid_rect.height()
        right, bottom = left + width, top + height
        center_x, center_y = self._grid_rect.center().x(), self._grid_rect.center().y()

        # --- 3. 绘制九宫格线 (垂直和水平三分线) ---
        one_third_x = left + width / 3
        two_thirds_x = left + 2 * width / 3
        painter.drawLine(QPointF(one_third_x, top), QPointF(one_third_x, bottom))
        painter.drawLine(QPointF(two_thirds_x, top), QPointF(two_thirds_x, bottom))
        
        one_third_y = top + height / 3
        two_thirds_y = top + 2 * height / 3
        painter.drawLine(QPointF(left, one_third_y), QPointF(right, one_third_y))
        painter.drawLine(QPointF(left, two_thirds_y), QPointF(right, two_thirds_y))

        # --- 4. 绘制对角线 (完成米字格) ---
        painter.drawLine(QPointF(left, top), QPointF(right, bottom))
        painter.drawLine(QPointF(right, top), QPointF(left, bottom))

        # --- 5. [核心新增] 绘制中文设计辅助线 (天地线) 和所有标签 ---
        
        # 定义标签颜色
        painter.setPen(QColor(Theme.LIGHT['content-secondary']))
        
        # a. 定义天地线和中线的相对位置 (0.0=顶, 1.0=底)
        chinese_guides = {
            "天": 0.125,
            "地": 0.875,
            # 水平中线已经由九宫格绘制，这里只为了添加标签
            "中线": 0.5,
        }
        
        # b. 绘制水平辅助线及其标签
        for name, ratio in chinese_guides.items():
            y = top + ratio * height
            # 只有天地线需要额外绘制，中线已经存在
            if name in ["天", "地"]:
                painter.setPen(pen) # 切换回虚线画笔
                painter.drawLine(QPointF(left, y), QPointF(right, y))
            
            painter.setPen(QColor(Theme.LIGHT['content-secondary'])) # 切换回文字画笔
            text_rect = QRectF(left - 80, y - 10, 70, 20)
            painter.drawText(text_rect, Qt.AlignRight | Qt.AlignVCenter, name)

        # c. 绘制垂直中线标签 (线本身已由九宫格绘制)
        text_rect = QRectF(center_x - 50, top - 22, 100, 20)
        painter.drawText(text_rect, Qt.AlignCenter, "中线")

        # d. 绘制框线标签
        # 上框线
        text_rect_top = QRectF(center_x - 50, top - 38, 100, 20)
        painter.drawText(text_rect_top, Qt.AlignCenter, "上框线")
        # 下框线
        text_rect_bottom = QRectF(center_x - 50, bottom + 5, 100, 20)
        painter.drawText(text_rect_bottom, Qt.AlignCenter, "下框线")
        # 左框线
        text_rect_left = QRectF(left - 80, center_y - 10, 70, 20)
        painter.drawText(text_rect_left, Qt.AlignRight | Qt.AlignVCenter, "左框线")
        # 右框线
        text_rect_right = QRectF(right + 10, center_y - 10, 70, 20)
        painter.drawText(text_rect_right, Qt.AlignLeft | Qt.AlignVCenter, "右框线")

    def _draw_raw_stroke(self, painter: QPainter, stroke: HandwritingStroke):
        """[私有辅助][已增强] 根据原始点和参数绘制笔画，用于实时预览。"""
        if len(stroke.points) < 2: return

        # 从 MainWindow 获取基础宽度
        base_width = self.main_window.stroke_width
        use_pressure = self.main_window.pressure_checkbox.isChecked()

        for i in range(len(stroke.points) - 1):
            p1_data, p2_data = stroke.points[i], stroke.points[i+1]
            
            p1_qt = QPointF(self._grid_rect.x() + p1_data[0] * self._pixel_size,
                            self._grid_rect.y() + p1_data[1] * self._pixel_size)
            p2_qt = QPointF(self._grid_rect.x() + p2_data[0] * self._pixel_size,
                            self._grid_rect.y() + p2_data[1] * self._pixel_size)
            
            width = base_width
            if use_pressure:
                # 使用两点的平均压力来决定线段宽度
                avg_pressure = (p1_data[2] + p2_data[2]) / 2.0
                width = max(1.0, base_width * avg_pressure)

            pen = QPen(QColor(stroke.color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(p1_qt, p2_qt)

    def _draw_vector_stroke(self, painter: QPainter, stroke: HandwritingStroke):
        """[私有辅助][已增强] 绘制经过矢量化并带有宽度的笔画。"""
        vector_path = stroke.to_bezier_path()
        if vector_path.is_empty(): return
        
        qpainter_path = vector_path.to_qpainter_path()
        
        transform = QTransform()
        transform.translate(self._grid_rect.x(), self._grid_rect.y())
        transform.scale(self._pixel_size, self._pixel_size)
        
        final_path = transform.map(qpainter_path)

        # [核心] 宽度现在也从 MainWindow 获取
        base_width = self.main_window.stroke_width
        pen = QPen(QColor(stroke.color), base_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(final_path)


    def _find_command_index_for_anchor(self, commands: List, anchor_idx: int) -> int:
        """找到包含指定索引锚点的命令的索引。"""
        counter = 0
        for i, cmd in enumerate(commands):
            if cmd[0] == 'moveTo':
                if counter == anchor_idx: return i
                counter += 1
            elif cmd[0] == 'qCurveTo':
                if counter == anchor_idx: return i
                counter += 1
        return -1
    def _get_node_pos_from_path(self, path: VectorPath, node_info: Dict[str, Any]) -> Optional[Point]:
        """
        [最终版 - 已修复直线段bug] 根据节点信息从VectorPath中提取其精确坐标。

        此版本修复了一个致命bug：原版缺少对 'lineTo' 命令的处理，导致
        由直线构成的路径中，只有第一个节点可以被正确识别和移动。
        现在，此函数能正确处理由任意 `moveTo`, `lineTo`, `curveTo` 组合成的路径。

        Args:
            path (VectorPath): 要在其中进行搜索的矢量路径对象。
            node_info (Dict[str, Any]): 一个描述要查找的节点的字典，
                例如: {'type': 'anchor', 'anchor_index': 2} 或
                      {'type': 'c_in', 'anchor_index': 2, 'seg_idx': 2}

        Returns:
            Optional[Point]: 如果找到，返回节点的 (x, y) 坐标元组；否则返回 None。
        """
        # --- 步骤 0: 从节点信息字典中解析所需参数 ---
        node_type = node_info.get('type')
        anchor_idx_target = node_info.get('anchor_index')

        # 如果关键信息缺失，则无法查找
        if node_type is None or anchor_idx_target is None:
            return None

        commands = path.commands
        
        # --- 步骤 1: 路径有效性检查 ---
        # 路径必须存在且以 'moveTo' 命令开始
        if not commands or commands[0][0] != 'moveTo':
            return None
        
        # --- 步骤 2: 处理第一个锚点 (特殊情况) ---
        # 第一个锚点（索引为0）总是 'moveTo' 命令的目标点，可以直接返回。
        if node_type == 'anchor' and anchor_idx_target == 0:
            return commands[0][1]

        # --- 步骤 3: 遍历后续命令查找目标节点 ---
        # 初始化锚点计数器，从1开始，因为索引0已经被处理
        current_anchor_idx = 1
        
        # 从第二个命令 (索引为1) 开始遍历
        for cmd in commands[1:]:
            
            # [核心修复] 新增对 'lineTo' 命令的处理
            if cmd[0] == 'lineTo':
                p1 = cmd[1]
                # 检查是否命中了当前锚点
                if node_type == 'anchor' and current_anchor_idx == anchor_idx_target:
                    return p1 # 找到了，返回该点的坐标
                
                # 如果没找到，增加锚点计数器，继续下一次循环
                current_anchor_idx += 1
            
            # 处理 'curveTo' 命令（保持原有逻辑）
            elif cmd[0] == 'curveTo':
                c1, c2, p1 = cmd[1], cmd[2], cmd[3]
                
                # a. 检查是否命中了上一个锚点的"出射句柄" (c_out)
                if node_type == 'c_out' and current_anchor_idx - 1 == anchor_idx_target:
                    return c1
                    
                # b. 检查是否命中了当前锚点的"入射句柄" (c_in)
                if node_type == 'c_in' and current_anchor_idx == anchor_idx_target:
                    return c2
                    
                # c. 检查是否命中了当前锚点本身 (anchor)
                if node_type == 'anchor' and current_anchor_idx == anchor_idx_target:
                    return p1
                
                # 如果当前段没有命中，则锚点索引加1，继续检查下一段
                current_anchor_idx += 1
                
        # --- 步骤 4: 如果遍历完所有命令都未找到，返回 None ---
        return None


    def _draw_professional_chinese_guides(self, painter: QPainter):
        """
        [私有辅助][新增] 绘制专业的中文字体设计辅助线，包括中宫、视觉重心和部件参考线。
        """
        # --- 1. 获取网格几何信息 ---
        left, top = self._grid_rect.left(), self._grid_rect.top()
        width, height = self._grid_rect.width(), self._grid_rect.height()
        right, bottom = left + width, top + height
        
        # --- 2. 绘制中宫 (Central Palace) ---
        zhonggong_scale = 0.78  # 中宫占字面框的比例，可调整
        inset_x = width * (1 - zhonggong_scale) / 2
        inset_y = height * (1 - zhonggong_scale) / 2
        zhonggong_rect = self._grid_rect.adjusted(inset_x, inset_y, -inset_x, -inset_y)
        
        pen_zhonggong = QPen(QColor("#e11d48"), 1, Qt.SolidLine) # 使用醒目的红色
        pen_zhonggong.setCosmetic(True) # 确保线条在任何缩放级别下都是1像素宽
        painter.setPen(pen_zhonggong)
        painter.drawRect(zhonggong_rect)
        
        # 绘制“中宫”标签
        painter.setFont(Theme.get_font("body"))
        label_rect = QRectF(zhonggong_rect.right() + 2, zhonggong_rect.top() - 8, 40, 20)
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, "中宫")

        # --- 3. 绘制视觉重心区域 (Visual Center) ---
        # 视觉重心通常在几何中心上方约 2-3% 的位置
        visual_center_y_offset = -height * 0.025
        visual_center = self._grid_rect.center() + QPointF(0, visual_center_y_offset)
        
        pen_center = QPen(QColor("#2563eb"), 1.5, Qt.SolidLine)
        pen_center.setCosmetic(True)
        painter.setPen(pen_center)
        # 绘制一个十字标记
        painter.drawLine(visual_center + QPointF(-5, 0), visual_center + QPointF(5, 0))
        painter.drawLine(visual_center + QPointF(0, -5), visual_center + QPointF(0, 5))

        # --- 4. 绘制部件结构参考线 (Component Lines) ---
        pen_component = QPen(QColor("#059669"), 1, Qt.DotLine) # 使用绿色点线
        pen_component.setCosmetic(True)
        painter.setPen(pen_component)

        # 左右结构参考线 (通常左部略窄)
        left_part_ratio = 0.45
        left_part_x = left + width * left_part_ratio
        painter.drawLine(QPointF(left_part_x, top), QPointF(left_part_x, bottom))

        # 上下结构参考线 (通常上部略窄)
        top_part_ratio = 0.48
        top_part_y = top + height * top_part_ratio
        painter.drawLine(QPointF(left, top_part_y), QPointF(right, top_part_y))



# ==============================================================================
# SECTION 4: 主窗口与控制器 (MAIN WINDOW & CONTROLLER)
#
# MainWindow 类是整个应用程序的核心，它扮演着视图(View)的容器和
# 控制器(Controller)的角色。
#
# 作为控制器，它：
# - 初始化并持有所有的模型对象 (如 FontDataManager) 和视图对象 (如 DrawingCanvas)。
# - 将视图的用户操作 (通过信号) 连接到自身的处理方法 (槽)。
# - 在处理方法中，调用模型的方法来更新数据。
# - 在模型数据更新后，调用视图的方法来刷新界面。
# ==============================================================================

class MainWindow(QMainWindow):
    """
    应用程序的主窗口。
    
    它扮演着视图(View)的容器和控制器(Controller)的角色。

    作为控制器，它：
    - 初始化并持有所有的模型对象 (如 FontDataManager) 和视图对象 (如 DrawingCanvas)。
    - 将视图的用户操作 (通过信号) 连接到自身的处理方法 (槽)。
    - 在处理方法中，调用模型的方法来更新数据。
    - 在模型数据更新后，调用视图的方法来刷新界面。
    """
    # ==========================================================================
    # ==== [新增] 内嵌SVG图标数据 ====
    # ==========================================================================
    SVG_ICONS = {
        'brush': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M14.16,3.83C14.55,3.44 15.18,3.44 15.57,3.83L19.17,7.43C19.56,7.82 19.56,8.45 19.17,8.84L10,18H6V14L14.16,3.83M18.17,6.43L16.57,4.83L14.5,6.91L16.1,8.5L18.17,6.43Z" />
          <path d="M5,20H19V22H5V20Z" />
        </svg>
        """,
        'pen': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M18.8,3.1C18.4,2.7 17.8,2.7 17.4,3.1L15,5.5L18.5,9L20.9,6.6C21.3,6.2 21.3,5.6 20.9,5.2L18.8,3.1M3,15.9L12.5,6.4L16.1,10L6.6,19.5L3,21L4.5,17.4L3,15.9Z" />
        </svg>
        """,
        'marker': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M18,14.4V11L21,7.9V5H3V7.9L6,11V14.4L3,17.5V20H21V17.5L18,14.4M16,12.5L15,11.5V8H9V11.5L8,12.5V14H16V12.5Z" />
        </svg>
        """,
        'pencil': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M14.06,9L15,9.94L5.92,19H5V18.08L14.06,9M17.66,3C17.41,3 17.15,3.1 16.96,3.29L15.13,5.12L18.88,8.87L20.71,7.04C21.1,6.65 21.1,6 20.71,5.63L18.37,3.29C18.17,3.09 17.92,3 17.66,3M14.06,6.19L3,17.25V21H6.75L17.81,9.94L14.06,6.19Z" />
        </svg>
        """,
        'eraser': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M16.24,3.56L21.19,8.5C21.97,9.29 21.97,10.55 21.19,11.34L12,20.53C10.44,22.09 7.91,22.09 6.34,20.53L2.81,17C1.25,15.44 1.25,12.91 2.81,11.35L11.66,2.5C12.45,1.71 13.71,1.71 14.5,2.5L16.24,4.22V3.56M10.25,6.63L7.5,9.38L12.75,14.63L15.5,11.88L10.25,6.63Z" />
        </svg>
        """,
        'node_editor': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M12.5,8C9.8,8 7.5,9.3 6.5,11.1L3.9,9.4C5.4,6.8 8.7,5 12.5,5C17.2,5 21.1,8 22,12.5C21.6,14.2 20.6,15.8 19.1,17.1L16.5,15.4C17.5,14.3 18,13.2 18,12C18,10.1 15.7,8 12.5,8M3,4.2L4.2,3L21,20.7L19.7,22L15.8,18.1C14.2,18.6 12.5,19 10.5,19C5.8,19 1.9,16 1,11.5C1.4,9.8 2.4,8.2 3.9,6.9L3,4.2M5.6,8.1C4.6,9.2 4,10.3 4,11.5C4.6,13.6 6.5,15.5 10.5,15.5C11.5,15.5 12.4,15.3 13.2,15L5.6,8.1Z" />
        </svg>
        """,
        'line': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M22,12L2,22L5.2,12L2,2L22,12Z" />
        </svg>
        """,
        'arc': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22V20A8,8 0 0,1 4,12A8,8 0 0,1 12,4V2Z" />
        </svg>
        """,
        'rect': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M4,4H20V20H4V4M6,6V18H18V6H6Z" />
        </svg>
        """,
        'circle': """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{color}">
          <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20Z" />
        </svg>
        """
    }
    # ==========================================================================
    # ==== 嵌套类：TextPreviewWidget 的正确位置 ====
    # 它必须是 MainWindow 类的直接成员，而不是某个方法的成员。
    # ==========================================================================
    class TextPreviewWidget(QWidget):
        """一个专门用于渲染文本行预览的自定义组件。"""
        def __init__(self, main_window: 'MainWindow', parent=None):
            super().__init__(parent)
            self.main_window = main_window
            self.setMinimumHeight(80)
            self.setStyleSheet("background-color: white; border: 1px solid #e2e8f0; border-radius: 4px;")
            self.text_to_render = ""

        def set_text(self, text: str):
            """设置需要渲染的文本并触发重绘。"""
            self.text_to_render = text
            self.update()

        def paintEvent(self, event: QPaintEvent):
            """使用 QPainter 逐个字符地绘制文本预览。"""
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            if not self.text_to_render:
                return

            current_x = 10
            char_size = self.height() - 20 # 字符高度
            
            for char_str in self.text_to_render:
                # 如果渲染超出了控件宽度，则停止
                if current_x > self.width() - char_size:
                    break

                # 从主窗口的数据模型中查找字符对象
                char_obj = self.main_window.font_chars.get(char_str)
                
                if char_obj and char_obj.is_designed:
                    # 如果字符已设计，调用其 get_preview_image 方法
                    pixmap = char_obj.get_preview_image(self.main_window.components, size=char_size)
                    # 使用字符的前进宽度来计算下一个字符的位置
                    advance_width = (char_obj.advance_width / char_obj.grid_size) * char_size
                else:
                    # 如果字符未设计或不存在，绘制一个系统字体占位符
                    pixmap = QPixmap(char_size, char_size)
                    pixmap.fill(Qt.transparent)
                    p = QPainter(pixmap)
                    font = QFont("Microsoft YaHei UI", int(char_size * 0.7))
                    p.setFont(font)
                    p.setPen(QColor("#94a3b8"))
                    p.drawText(pixmap.rect(), Qt.AlignCenter, char_str)
                    p.end()
                    advance_width = char_size
                
                painter.drawPixmap(QPoint(int(current_x), 10), pixmap)
                current_x += advance_width
    class KerningPreviewWidget(QWidget):
        """一个专门用于渲染和实时调整字偶距的自定义预览组件。"""
        def __init__(self, main_window: 'MainWindow', parent=None):
            super().__init__(parent)
            self.main_window = main_window
            self.setMinimumHeight(120)
            self.setStyleSheet("background-color: white; border: 1px solid #e2e8f0; border-radius: 4px;")
            self.pair_text = "你我"
            self.kerning_value = 0

        def set_pair(self, text: str, value: int):
            """设置需要渲染的字符对和字偶距值，并触发重绘。"""
            self.pair_text = text[:2] # 只取前两个字符
            self.kerning_value = value
            self.update()

        def paintEvent(self, event: QPaintEvent):
            """使用 QPainter 逐个字符地绘制并应用字偶距。"""
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            if len(self.pair_text) < 2:
                painter.setPen(QColor("#9ca3af"))
                painter.drawText(self.rect(), Qt.AlignCenter, "请输入至少两个字符")
                return

            char_size = self.height() - 40 # 字符高度
            
            # 垂直居中
            y_pos = (self.height() - char_size) / 2
            
            # 水平居中计算
            total_width = 0
            pixmaps = []
            advances = []
            for char_str in self.pair_text:
                char_obj = self.main_window.font_chars.get(char_str)
                pixmap = QPixmap(char_size, char_size)
                pixmap.fill(Qt.transparent)
                p = QPainter(pixmap)
                advance = char_size
                if char_obj and char_obj.is_designed:
                    p.drawPixmap(0, 0, char_obj.get_preview_image(self.main_window.components, size=char_size))
                    advance = (char_obj.advance_width / char_obj.grid_size) * char_size
                else:
                    font = QFont("Microsoft YaHei UI", int(char_size * 0.7))
                    p.setFont(font); p.setPen(QColor("#9ca3af"))
                    p.drawText(pixmap.rect(), Qt.AlignCenter, char_str)
                p.end()
                pixmaps.append(pixmap)
                advances.append(advance)
                total_width += advance
            
            # 应用字偶距
            kerning_px = (self.kerning_value / self.main_window.grid_size) * char_size
            total_width += kerning_px

            current_x = (self.width() - total_width) / 2

            # 绘制第一个字符
            painter.drawPixmap(QPoint(int(current_x), int(y_pos)), pixmaps[0])
            current_x += advances[0]
            
            # 应用字偶距并绘制第二个字符
            current_x += kerning_px
            painter.drawPixmap(QPoint(int(current_x), int(y_pos)), pixmaps[1])
    # ==========================================================================
    # ==== MainWindow 的常规方法从这里开始 ====
    # ==========================================================================           
    def __init__(self):
        """
        [控制器][V5.7 字偶距功能增强版] 初始化主窗口。
        
        此版本新增了 `self.kerning_pairs` 字典作为字偶距数据模型，
        为实现完整的字偶距调整功能奠定了数据基础。
        """
        super().__init__()
        
        # --- 步骤 1: 初始化模型数据和状态 ---
        self.data_manager = FontDataManager()
        self.font_chars: Dict[str, FontChar] = {}
        self.components: Dict[str, FontComponent] = {}
        
        # 核心状态变量
        self.current_char_obj: Optional[FontChar] = None
        self.is_project_dirty: bool = False
        self.current_project_path: Optional[str] = None
        
        # [核心新增] --- 字偶距数据模型 ---
        self.kerning_pairs: Dict[str, int] = {}
        
        # 字体设置数据模型
        self.font_settings = {
            'familyName': "我的手写字体",
            'styleName': "常规",
            'version': "1.000",
            'copyright': "© Your Name",
            'unitsPerEm': 1024,
            'ascender': 880,
            'descender': -120,
            'xHeight': 480,
            'capHeight': 700,
        }

        # 撤销/重做系统
        self.undo_stack = QUndoStack(self)

        # 工具状态变量
        self.grid_size: int = 1024
        self.current_tool: str = "brush"
        self.current_color: QColor = QColor("#2d3748")
        self.stroke_width: int = 4
        self.stroke_smoothing: float = 0.4
        
        # --- 步骤 2: 预先声明所有将被引用的UI组件属性 ---
        self.navbar: Optional[QWidget] = None
        self.status_label: Optional[QLabel] = None
        self.canvas: Optional[DrawingCanvas] = None
        self.header_char_display: Optional[QLabel] = None
        self.header_char_label: Optional[QLabel] = None
        self.header_char_info_label: Optional[QLabel] = None
        self.tool_button_group: Optional[QButtonGroup] = None
        
        self.tool_presets = {
            'brush':    {'name': '毛笔', 'base_smoothing': 0.6, 'pressure_effect': 1.0,  'rdp_epsilon_factor': 0.8},
            'pen':      {'name': '钢笔', 'base_smoothing': 0.2, 'pressure_effect': 0.3,  'rdp_epsilon_factor': 1.5},
            'marker':   {'name': '马克笔', 'base_smoothing': 0.1, 'pressure_effect': 0.0,  'rdp_epsilon_factor': 2.0},
            'pencil':   {'name': '铅笔', 'base_smoothing': 0.8, 'pressure_effect': 0.7,  'rdp_epsilon_factor': 0.5},
            'eraser':   {'name': '橡皮擦', 'base_smoothing': 0.0, 'pressure_effect': 0.0,  'rdp_epsilon_factor': 3.0},
            'node_editor': {'name': '节点编辑'}
        }
        
        # --- 步骤 3: 设置窗口基本属性 ---
        self.setWindowTitle("MCDCNFD 字体设计  作者：跳舞的火公子 (PyQt5 UI 还原版)")
        self.setGeometry(100, 100, 1800, 1000)
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(Theme.get_qss())

        # --- 步骤 4: 创建并组织UI ---
        self._create_actions()
        self._create_custom_toolbar()
        self._create_statusbar()
        self._create_central_widget()
        self._create_docks()
        
        # --- 步骤 5: 连接所有信号与槽 ---
        self._connect_signals()

        # --- 步骤 6: 启动数据加载和最终初始化 ---
        QTimer.singleShot(0, self._startup)
    
    def _startup(self):
        """执行应用启动时的初始化任务。"""
        try:
            self.statusBar().showMessage("正在初始化基础数据库...")
            self.data_manager.initialize()
            self.load_project_data("font_char_data.db", is_db=True)
            self._init_tool_state()
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"加载数据时发生严重错误: {e}")
            self.close()

    # --- UI 创建辅助函数 ---
    def _create_actions(self):
        """创建所有功能对应的 QAction 对象。"""
        self.load_db_action = QAction("加载字库...", self, statusTip="从数据库文件开始一个新项目")
        self.new_action = QAction("新建项目", self, statusTip="清空当前项目，基于已加载的字库开始新设计")
        self.open_action = QAction("打开项目...", self, statusTip="打开一个之前保存的 .mcdcnfd 项目文件")
        self.save_action = QAction("保存项目", self, shortcut="Ctrl+S", statusTip="保存当前项目")
        self.save_as_action = QAction("另存为...", self, shortcut="Ctrl+Shift+S", statusTip="将当前项目另存为新文件")
        self.export_ttf_action = QAction("导出 TTF...", self, statusTip="将设计的字符导出为TTF字体文件")
        self.exit_action = QAction("退出", self, shortcut="Ctrl+Q", statusTip="退出应用程序")
        
        self.undo_action = self.undo_stack.createUndoAction(self, "撤销")
        self.undo_action.setShortcut("Ctrl+Z")
        self.redo_action = self.undo_stack.createRedoAction(self, "重做")
        self.redo_action.setShortcut("Ctrl+Y")

    def _create_custom_toolbar(self):
        """
        [最终优雅版] 创建一个自定义QToolBar，并使用QSS来设置大标题样式。
        """
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setStyleSheet(f"QToolBar {{ background-color: {Theme.LIGHT['accent-primary']}; padding: 0px; border: none; }}")
        
        navbar_widget = QWidget()
        # 仍然建议保留一个最小高度，以确保在所有平台和样式下都有稳定的布局
        navbar_widget.setMinimumHeight(60)
        
        layout = QHBoxLayout(navbar_widget)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(10)
        
        title_label = QLabel("MCDCNFD 字体设计  作者：跳舞的火公子")
        
        # --- [核心修改] ---
        # 只需设置对象名，所有样式都将由全局QSS自动应用
        title_label.setObjectName("NavTitleLabel")
        
        layout.addWidget(title_label)
        layout.addStretch()

        # --- 按钮逻辑保持不变 ---
        actions = [
            self.load_db_action, self.new_action, self.open_action, self.save_action, 
            self.save_as_action, self.export_ttf_action
        ]
        for action in actions:
            btn = QPushButton(action.text().replace('&',''))
            btn.clicked.connect(action.trigger)
            btn.setStatusTip(action.statusTip())
            btn.setStyleSheet(f"""
                QPushButton {{ 
                    background-color: {Theme.LIGHT['accent-primary-hover']};
                    color: white; padding: 8px 14px; border-radius: 4px;
                    font-weight: bold;
                }}
                QPushButton:hover {{ background-color: #7c74f2; }}
                QPushButton:pressed {{ background-color: {Theme.LIGHT['accent-primary-pressed']}; }}
            """)
            layout.addWidget(btn)

        toolbar.addWidget(navbar_widget)
        self.addToolBar(toolbar)
    def _create_statusbar(self):
        """创建状态栏。"""
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label)
        
    def _create_central_widget(self):
        """
        [已修正] 创建包含标题区和画布的中央区域。
        
        此版本修正了 DrawingCanvas 的实例化方式，确保将主窗口的
        引用 (self) 传递给它，以便画布能够访问全局数据模型，
        例如用于绘制度量线的 font_metadata。
        """
        # --- 1. 创建主容器和布局 ---
        central_container = QWidget()
        layout = QVBoxLayout(central_container)
        layout.setContentsMargins(0, 0, 0, 0) # 容器自身无外边距
        layout.setSpacing(0) # 标题区和画布区紧密相连
        
        # --- 2. 创建并添加中央面板的标题头 ---
        # 调用辅助方法来构建复杂的标题区
        header = self._create_central_header()
        layout.addWidget(header)

        # --- 3. 创建画布区域 ---
        # 使用一个带内边距的 Frame 作为画布的“画框”，提供视觉上的呼吸空间
        canvas_frame = QFrame()
        canvas_frame.setObjectName("CanvasFrame")
        canvas_frame.setStyleSheet(
            f"#CanvasFrame {{ "
            f"border: none; "
            f"background-color: {Theme.LIGHT['bg-tertiary']}; "
            f"padding: 20px; " # 画布与中央区域边缘的间距
            f"}}"
        )
        canvas_layout = QVBoxLayout(canvas_frame)
        
        # *** 关键修正: 更新 DrawingCanvas 的实例化 ***
        # 按照新的构造函数 DrawingCanvas(main_window, parent) 进行调用。
        # - main_window=self: 将主窗口实例的引用传递给画布。
        # - parent=canvas_frame: 将画布的父组件设置为我们刚创建的画框。
        self.canvas = DrawingCanvas(main_window=self, parent=canvas_frame)
        
        # [移除] 不再需要手动设置 grid_size，因为它会从 main_window 自动获取。
        # self.canvas.grid_size = self.grid_size
        
        canvas_layout.addWidget(self.canvas)
        
        # 将包含画布的画框添加到主布局，并设置拉伸因子为1，使其填充所有可用垂直空间
        layout.addWidget(canvas_frame, 1)
        
        # --- 4. 将最终完成的中央面板设置为主窗口的中央控件 ---
        self.setCentralWidget(central_container)
    def _create_central_header(self) -> QWidget:
        """
        [已修正] 创建中央面板顶部的标题区，并正确设置实例属性。
        """
        header_widget = QFrame()
        header_widget.setFixedHeight(80)
        header_widget.setStyleSheet(f"background-color: {Theme.LIGHT['bg-secondary']}; border-bottom: 1px solid {Theme.LIGHT['border-primary']};")
        
        layout = QHBoxLayout(header_widget)
        layout.setContentsMargins(20, 0, 20, 0)

        self.header_char_display = QLabel('')
        self.header_char_display.setFixedSize(60, 60); self.header_char_display.setAlignment(Qt.AlignCenter)
        self.header_char_display.setFont(QFont("Microsoft YaHei UI", 30, QFont.Bold))
        self.header_char_display.setStyleSheet(f"background-color: {Theme.LIGHT['accent-primary']}; color: white; border-radius: 8px;")
        layout.addWidget(self.header_char_display)

        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(15, 0, 0, 0); info_layout.setSpacing(0)
        
        # [样式还原] 使用更合适的标题字体
        self.header_char_label = QLabel('请选择一个字符开始编辑')
        self.header_char_label.setFont(Theme.get_font("h3")) # <-- 修改在这里
        
        self.header_char_info_label = QLabel('Unicode: | 笔画数: | 拼音:')
        self.header_char_info_label.setStyleSheet(f"color: {Theme.LIGHT['content-secondary']};")
        
        info_layout.addStretch(); info_layout.addWidget(self.header_char_label)
        info_layout.addWidget(self.header_char_info_label); info_layout.addStretch()
        
        layout.addWidget(info_widget, 1) 
        
        return header_widget
    def _create_docks(self):
        """
        [已增强] 创建所有可停靠的侧边栏 (Dock Widgets) 并进行样式化。
        
        此版本新增了对“属性”标签页的调用，该标签页用于显示和编辑
        当前选中的节点或线段的详细信息。
        """
        # 允许 Dock 之间有动画效果和嵌套
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks)
        
        # --- 步骤 1: 创建左侧 Dock ---
        left_dock = QDockWidget("浏览器与工具", self)
        left_dock.setTitleBarWidget(QWidget()) # 关键：隐藏默认标题栏
        left_dock.setFeatures(QDockWidget.NoDockWidgetFeatures) # 禁止关闭、浮动
        left_dock.setFixedWidth(350) # 设置固定宽度
        
        # 在 Dock 内部创建一个标签页控件
        left_tab_widget = QTabWidget()
        left_dock.setWidget(left_tab_widget)
        
        # 调用辅助方法填充左侧的标签页内容
        self._create_char_browser_tab(left_tab_widget)
        self._create_design_tab(left_tab_widget)
        
        # 将创建好的 Dock 添加到主窗口的左侧区域
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # --- 步骤 2: 创建右侧 Dock ---
        right_dock = QDockWidget("属性与预览", self)
        right_dock.setTitleBarWidget(QWidget()) # 关键：隐藏默认标题栏
        right_dock.setFeatures(QDockWidget.NoDockWidgetFeatures) # 禁止关闭、浮动
        right_dock.setFixedWidth(380) # 设置固定宽度
        
        # 在 Dock 内部创建一个标签页控件
        right_tab_widget = QTabWidget()
        right_dock.setWidget(right_tab_widget)
        
        # 调用辅助方法填充右侧的所有标签页内容
        self._create_preview_tab(right_tab_widget)
        self._create_analysis_tab(right_tab_widget)
        
        # [核心新增] 调用新的方法来创建“属性”标签页
        self._create_properties_tab(right_tab_widget)
        
        self._create_settings_tab(right_tab_widget)
        self._create_export_tab(right_tab_widget)
        self._create_history_tab(right_tab_widget)
        self._create_kerning_tab(right_tab_widget)
        
        # 将创建好的 Dock 添加到主窗口的右侧区域
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

    def _create_properties_tab(self, parent_tab_widget: QTabWidget):
        """
        [新增] 创建并填充“属性”标签页，用于显示和编辑选中对象的属性。
        
        此面板是上下文感知的，会根据用户在画布上的选择（节点、线段等）
        动态地显示不同的信息和编辑控件。
        """
        # --- 步骤 1: 创建该标签页的主容器和布局 ---
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # --- 步骤 2: 创建“当前选择”信息卡片 ---
        # 这个卡片始终可见，用于显示选中对象的基本信息。
        info_card = CardWidget("当前选择", theme=Theme.LIGHT)
        info_layout = QGridLayout(info_card.contentWidget())
        info_layout.setContentsMargins(12, 12, 12, 12)
        
        # 创建用于显示信息的标签，并设置为实例属性以便后续更新
        self.prop_selection_type_label = QLabel("无")
        self.prop_length_label = QLabel("N/A")
        self.prop_angle_label = QLabel("N/A")
        
        # 将标签和其标题添加到布局中
        info_layout.addWidget(QLabel("类型:"), 0, 0)
        info_layout.addWidget(self.prop_selection_type_label, 0, 1)
        info_layout.addWidget(QLabel("长度/弧长:"), 1, 0)
        info_layout.addWidget(self.prop_length_label, 1, 1)
        info_layout.addWidget(QLabel("角度:"), 2, 0)
        info_layout.addWidget(self.prop_angle_label, 2, 1)
        
        layout.addWidget(info_card)

        # --- 步骤 3: 创建“线段属性”卡片 (默认隐藏) ---
        # 这个卡片只在线段被选中时才显示。
        self.segment_props_card = CardWidget("线段属性", theme=Theme.LIGHT)
        segment_layout = QGridLayout(self.segment_props_card.contentWidget())
        segment_layout.setContentsMargins(12, 12, 12, 12)
        
        # 创建张力滑块和对应的百分比标签
        self.segment_tension_slider = QSlider(Qt.Horizontal)
        self.segment_tension_slider.setRange(0, 200) # 允许张力在 0% 到 200% 之间调整
        self.segment_tension_slider.setValue(100)    # 默认值为 100%
        self.segment_tension_label = QLabel("100%")
        
        segment_layout.addWidget(QLabel("张力:"), 0, 0)
        segment_layout.addWidget(self.segment_tension_slider, 0, 1)
        segment_layout.addWidget(self.segment_tension_label, 0, 2)
        
        layout.addWidget(self.segment_props_card)
        
        # --- 步骤 4: 创建“节点属性”卡片 (默认隐藏) ---
        # 预留位置，为未来的节点属性（如可变宽度）做准备
        self.node_props_card = CardWidget("节点属性", theme=Theme.LIGHT)
        # ... (未来可以在这里添加节点相关的控件，例如 QSlider for width factor) ...
        layout.addWidget(self.node_props_card)

        # 添加一个伸缩项，将所有卡片推到顶部
        layout.addStretch()
        
        # 将最终完成的 widget 添加到父标签页
        parent_tab_widget.addTab(widget, "📝 属性")

        # --- 初始状态设置 ---
        # 默认情况下，特定于选择类型的属性卡片是隐藏的
        self.segment_props_card.hide()
        self.node_props_card.hide()

    def _update_properties_panel(self):
        """
        [V1.2 已修复IndexError] 根据当前画布的选择，动态更新属性面板的内容和可见性。
        
        此版本通过增加对选中笔画索引的严格范围检查，彻底解决了在切换字符
        或删除笔画后可能发生的 `IndexError`。
        """
        canvas = self.canvas
        
        # --- 步骤 1: 默认状态重置 ---
        # 在每次更新前，先将面板恢复到最基础的“无选择”状态。
        self.prop_selection_type_label.setText("无")
        self.prop_length_label.setText("N/A")
        self.prop_angle_label.setText("N/A")
        self.segment_props_card.hide()
        self.node_props_card.hide()

        # [核心修复] 增加对 self.current_char_obj 的严格检查，并确保
        # selected_stroke_index 是一个在当前 strokes 列表中的有效索引。
        if not (canvas and self.current_char_obj and 
                0 <= canvas.selected_stroke_index < len(self.current_char_obj.strokes)):
            # 如果不满足以上任何一个条件，说明当前没有一个“有效的”选中笔画，
            # 必须立即返回，不能继续执行。
            return

        # --- 步骤 2: 获取基础数据 ---
        # 经过上面的严格检查，现在可以安全地访问 strokes 属性
        stroke = self.current_char_obj.strokes[canvas.selected_stroke_index]
        path = stroke.to_bezier_path()

        # --- 步骤 3: 处理线段选择 ---
        if canvas.selected_segment_index is not None:
            seg_idx = canvas.selected_segment_index
            if 0 < seg_idx < len(path.commands):
                cmd = path.commands[seg_idx]
                p0 = canvas._get_anchor_point_before(path.commands, seg_idx)
                p1 = None
                
                # 确定线段的终点
                if cmd[0] == 'lineTo': p1 = cmd[1]
                elif cmd[0] == 'curveTo': p1 = cmd[3]
                elif cmd[0] == 'closePath': p1 = canvas._get_anchor_point_before(path.commands, 1)

                if p0 and p1:
                    # a. 更新通用信息卡片 (长度和角度)
                    chord_length = math.dist(p0, p1)
                    angle = math.degrees(math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0])) % 360
                    
                    self.prop_angle_label.setText(f"{angle:.2f}°")

                    # b. 根据线段类型显示特定信息
                    if cmd[0] == 'curveTo':
                        self.prop_selection_type_label.setText(f"曲线段 (索引 {seg_idx})")
                        
                        arc_length = path._get_bezier_length(p0, cmd[1], cmd[2], p1, num_segments=50, bezier_type='cubic')
                        self.prop_length_label.setText(f"{arc_length:.2f} px")
                        
                        self.segment_props_card.show()
                        
                        c1 = cmd[1]
                        dist_handle = math.dist(c1, p0)
                        tension = (dist_handle / chord_length) * 100 if chord_length > 1e-6 else 100
                        
                        self.segment_tension_slider.blockSignals(True)
                        self.segment_tension_slider.setValue(int(tension))
                        self.segment_tension_label.setText(f"{int(tension)}%")
                        self.segment_tension_slider.blockSignals(False)
                        
                    else: # lineTo or closePath
                        self.prop_selection_type_label.setText(f"直线段 (索引 {seg_idx})")
                        self.prop_length_label.setText(f"{chord_length:.2f} px")

        # --- 步骤 4: 处理节点选择 ---
        elif canvas.selected_anchor_indices:
            num_selected = len(canvas.selected_anchor_indices)
            self.prop_selection_type_label.setText(f"{num_selected} 个节点")
            # self.node_props_card.show() # 为未来功能预留
    
    
    def _create_char_browser_tab(self, parent_tab_widget: QTabWidget):
        """
        [已修正] 创建并填充“字符浏览”标签页，布局更接近原版。
        
        此版本修正了因 CardWidget 重构导致的 AttributeError，
        确保为每个 CardWidget 的 contentWidget 显式创建和设置布局。
        """
        # --- 1. 创建该标签页的主容器和顶层布局 ---
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)
        
        # --- 2. 顶层标题和刷新按钮 ---
        header_layout = QHBoxLayout()
        header_label = QLabel("智能字符浏览器")
        header_label.setFont(Theme.get_font("title")) # 使用更合适的字体
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._perform_search)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(refresh_btn)
        layout.addLayout(header_layout)

        # --- 3. 搜索框和分类下拉框 ---
        search_layout = QGridLayout()
        search_layout.setSpacing(10)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("搜索字符、拼音、部首...")
        self.category_combo = QComboBox()
        
        search_layout.addWidget(QLabel("字符分类:"), 0, 0)
        search_layout.addWidget(self.category_combo, 0, 1)
        search_layout.addWidget(self.search_box, 1, 0, 1, 2)
        layout.addLayout(search_layout)

        # --- 4. 高级过滤卡片 ---
        filter_card = CardWidget("高级过滤", theme=Theme.LIGHT)
        
        # *** 关键修正: 为 filter_card 的 contentWidget 创建布局 ***
        filter_layout = QGridLayout(filter_card.contentWidget())
        filter_layout.setContentsMargins(12, 12, 12, 12) # 设置内边距
        
        # 笔画数范围
        stroke_range_layout = QHBoxLayout()
        self.min_strokes_spinbox = QSpinBox(); self.min_strokes_spinbox.setRange(0, 100); self.min_strokes_spinbox.setValue(0)
        self.max_strokes_spinbox = QSpinBox(); self.max_strokes_spinbox.setRange(0, 100); self.max_strokes_spinbox.setValue(40)
        stroke_range_layout.addWidget(self.min_strokes_spinbox)
        stroke_range_layout.addWidget(QLabel("至"))
        stroke_range_layout.addWidget(self.max_strokes_spinbox)
        
        # 设计状态
        status_layout = QHBoxLayout()
        self.designed_checkbox = QCheckBox("已设计"); self.designed_checkbox.setChecked(True)
        self.undesigned_checkbox = QCheckBox("未设计"); self.undesigned_checkbox.setChecked(True)
        status_layout.addWidget(self.designed_checkbox); status_layout.addWidget(self.undesigned_checkbox); status_layout.addStretch()

        # 现在使用我们新创建的 filter_layout 来添加控件
        filter_layout.addWidget(QLabel("笔画数:"), 0, 0)
        filter_layout.addLayout(stroke_range_layout, 0, 1)
        filter_layout.addWidget(QLabel("设计状态:"), 1, 0)
        filter_layout.addLayout(status_layout, 1, 1)
        layout.addWidget(filter_card)

        # --- 5. 参考底模卡片 ---
        ref_card = CardWidget("参考底模", theme=Theme.LIGHT)
        
        # *** 关键修正: 为 ref_card 的 contentWidget 创建布局 ***
        ref_card_layout = QVBoxLayout(ref_card.contentWidget())
        ref_card_layout.setContentsMargins(12, 12, 12, 12) # 设置内边距

        self.char_as_ref_checkbox = QCheckBox("显示当前字符的标准字形作为底模")
        
        # 现在使用我们新创建的 ref_card_layout 来添加控件
        ref_card_layout.addWidget(self.char_as_ref_checkbox)
        layout.addWidget(ref_card)

        # --- 6. 结果计数和分页信息 ---
        result_info_layout = QHBoxLayout()
        self.char_count_label = QLabel("找到 0 个")
        self.page_info_label = QLabel("第 1 / 1 页")
        result_info_layout.addWidget(self.char_count_label)
        result_info_layout.addStretch()
        result_info_layout.addWidget(self.page_info_label)
        layout.addLayout(result_info_layout)

        # --- 7. 高性能字符列表视图 ---
        self.char_list_view = QListView()
        self.char_list_view.setViewMode(QListView.IconMode)
        self.char_list_view.setResizeMode(QListView.Adjust)
        self.char_list_view.setMovement(QListView.Static)
        self.char_list_view.setGridSize(QSize(58, 64))
        self.char_list_view.setSpacing(6)
        self.char_list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.char_list_model = CharListModel()
        self.char_list_view.setModel(self.char_list_model)
        self.char_item_delegate = CharItemDelegate(self.char_list_view, self)
        self.char_list_view.setItemDelegate(self.char_item_delegate)
        layout.addWidget(self.char_list_view, 1)

        # --- 8. 分页按钮 ---
        page_btn_layout = QHBoxLayout()
        self.prev_page_btn = QPushButton("上一页")
        self.next_page_btn = QPushButton("下一页")
        page_btn_layout.addWidget(self.prev_page_btn)
        page_btn_layout.addWidget(self.next_page_btn)
        layout.addLayout(page_btn_layout)
        
        # --- 9. 将最终的 widget 添加到父标签页 ---
        parent_tab_widget.addTab(widget, "🔍 浏览")
    def _create_design_tab(self, parent_tab_widget: QTabWidget):
        """
        [最终UI还原版] 创建并填充“设计”标签页。
        
        此版本精确还原了原始UI的二级标签页结构，将“工具”、“图层”、
        “部件”和“AI助手”组织在了一个子 QTabWidget 中。
        """
        # --- 1. 创建该标签页的主容器 ---
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # --- 2. 创建并添加子标签页控件 ---
        design_tabs = QTabWidget()
        layout.addWidget(design_tabs)
        
        # --- 3. 调用辅助方法，填充每一个子标签页 ---
        self._populate_tools_sub_tab(design_tabs)
        self._populate_layers_sub_tab(design_tabs)
        self._populate_components_sub_tab(design_tabs)
        self._populate_ai_sub_tab(design_tabs)
        
        # --- 4. 将最终的 widget 添加到主标签页 ---
        parent_tab_widget.addTab(widget, "🎨 设计")

    def _populate_tools_sub_tab(self, parent_tab_widget: QTabWidget):
        """
        [V2.2 - 节点类型增强版] 填充“设计”->“工具”子标签页的内容。

        此版本根据“层次一：核心功能补完”方案，在“编辑操作”面板中
        增加了“转为非对称”按钮，并将其完全整合到UI布局和按钮管理逻辑中。
        """
        # --- 步骤 1: 创建该子标签页的主容器和布局 ---
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(15)

        # --- 步骤 2: 创建“绘图工具” QGroupBox ---
        tools_group = QGroupBox("绘图工具")
        tools_grid = QGridLayout(tools_group)
        
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)

        self.tool_buttons: Dict[str, QPushButton] = {}
        tools_with_icons = [
            ('毛笔', 'brush'), ('钢笔', 'pen'), ('马克笔', 'marker'), 
            ('铅笔', 'pencil'), ('橡皮擦', 'eraser'), ('节点编辑', 'node_editor'),
            ('直线', 'line'), ('弧线', 'arc'), ('矩形', 'rect'), ('圆形', 'circle')
        ]
        
        for i, (name, tool_id) in enumerate(tools_with_icons):
            btn = QPushButton(name)
            svg_data_template = self.SVG_ICONS.get(tool_id)
            if svg_data_template:
                svg_data_normal = svg_data_template.format(color=Theme.LIGHT['content-primary'])
                pixmap_normal = QPixmap(); pixmap_normal.loadFromData(svg_data_normal.encode('utf-8'))
                svg_data_checked = svg_data_template.format(color=Theme.LIGHT['accent-on-primary'])
                pixmap_checked = QPixmap(); pixmap_checked.loadFromData(svg_data_checked.encode('utf-8'))
                icon = QIcon(); icon.addPixmap(pixmap_normal, QIcon.Normal, QIcon.Off); icon.addPixmap(pixmap_checked, QIcon.Normal, QIcon.On)
                btn.setIcon(icon); btn.setIconSize(QSize(20, 20))
            
            btn.setCheckable(True)
            btn.setProperty("toolButton", "true")
            btn.setProperty("tool_id", tool_id)
            self.tool_buttons[tool_id] = btn
            tools_grid.addWidget(btn, i // 2, i % 2)
            self.tool_button_group.addButton(btn)
        layout.addWidget(tools_group)

        # --- 步骤 3: 创建“笔画参数” QGroupBox ---
        params_group = QGroupBox("笔画参数")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QLabel("笔触宽度:"), 0, 0)
        self.stroke_width_slider = QSlider(Qt.Horizontal); self.stroke_width_slider.setRange(1, 50)
        params_layout.addWidget(self.stroke_width_slider, 0, 1)
        self.stroke_width_label = QLabel("4"); self.stroke_width_label.setMinimumWidth(25)
        params_layout.addWidget(self.stroke_width_label, 0, 2)
        params_layout.addWidget(QLabel("平滑度:"), 1, 0)
        self.smoothing_slider = QSlider(Qt.Horizontal); self.smoothing_slider.setRange(0, 100)
        params_layout.addWidget(self.smoothing_slider, 1, 1)
        self.smoothing_label = QLabel("0.4"); self.smoothing_label.setMinimumWidth(25)
        params_layout.addWidget(self.smoothing_label, 1, 2)
        params_layout.addWidget(QLabel("笔画颜色:"), 2, 0)
        self.color_btn = QPushButton(); self.color_btn.setFixedSize(32, 32)
        params_layout.addWidget(self.color_btn, 2, 1, alignment=Qt.AlignLeft)
        self.pressure_checkbox = QCheckBox("启用压感效果")
        self.antialias_checkbox = QCheckBox("抗锯齿 (预览)"); self.antialias_checkbox.setChecked(True)
        params_layout.addWidget(self.pressure_checkbox, 3, 0, 1, 3)
        params_layout.addWidget(self.antialias_checkbox, 4, 0, 1, 3)
        layout.addWidget(params_group)
        
        # --- 步骤 4: 创建“显示选项” QGroupBox ---
        display_options_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_options_group)
        self.show_grid_checkbox = QCheckBox("显示网格"); self.show_grid_checkbox.setChecked(True)
        self.show_guides_checkbox = QCheckBox("显示米字格/天地线"); self.show_guides_checkbox.setChecked(True)
        self.show_metrics_checkbox = QCheckBox("显示西文度量线"); self.show_metrics_checkbox.setChecked(True)
        
        self.show_pro_guides_checkbox = QCheckBox("显示专业中文辅助线 (中宫/重心)")
        self.show_pro_guides_checkbox.setChecked(True)

        display_layout.addWidget(self.show_grid_checkbox)
        display_layout.addWidget(self.show_guides_checkbox)
        display_layout.addWidget(self.show_metrics_checkbox)
        display_layout.addWidget(self.show_pro_guides_checkbox)
        
        layout.addWidget(display_options_group)
        
        # --- 步骤 5: [核心修改] 创建“编辑操作” QGroupBox ---
        actions_group = QGroupBox("编辑操作")
        actions_layout = QGridLayout(actions_group)
        
        # a. 创建所有按钮
        self.undo_btn = QPushButton("撤销")
        self.redo_btn = QPushButton("重做")
        self.clear_char_btn = QPushButton("清除")
        self.vectorize_glyph_btn = QPushButton("字形矢量化")
        self.vectorize_glyph_btn.setToolTip("将当前字符的标准字形轮廓转换为矢量笔画")
        
        self.to_corner_btn = QPushButton("转为尖角")
        self.to_smooth_btn = QPushButton("转为平滑")
        # [核心新增] 创建“转为非对称”按钮
        self.to_asymmetric_btn = QPushButton("转为非对称")
        
        self.close_path_btn = QPushButton("闭合路径")
        self.break_path_btn = QPushButton("断开路径")
        self.insert_node_btn = QPushButton("插入节点")
        self.insert_node_btn.setCheckable(True)
        self.delete_node_btn = QPushButton("删除节点")
        self.merge_nodes_btn = QPushButton("合并节点")
        self.curve_to_line_btn = QPushButton("曲线转直线")
        self.line_to_curve_btn = QPushButton("直线转曲线")

        # b. 将所有新按钮放入一个字典中方便管理
        self.node_action_buttons = {
            "to_corner": self.to_corner_btn, 
            "to_smooth": self.to_smooth_btn,
            # [核心新增] 将新按钮加入字典
            "to_asymmetric": self.to_asymmetric_btn,
            "close_path": self.close_path_btn, "break_path": self.break_path_btn,
            "insert_node": self.insert_node_btn, "delete_node": self.delete_node_btn,
            "merge_nodes": self.merge_nodes_btn, "curve_to_line": self.curve_to_line_btn,
            "line_to_curve": self.line_to_curve_btn
        }
        # 默认禁用所有节点操作按钮
        for btn in self.node_action_buttons.values():
            btn.setEnabled(False)

        # c. 设置按钮样式
        self.undo_btn.setStyleSheet(f"background-color: {Theme.LIGHT['warning']};")
        self.redo_btn.setStyleSheet(f"background-color: {Theme.LIGHT['info']};")
        self.clear_char_btn.setStyleSheet(f"background-color: {Theme.LIGHT['danger']};")
        self.vectorize_glyph_btn.setStyleSheet(f"background-color: {Theme.LIGHT['success']};")
        
        # d. 将按钮添加到网格布局
        actions_layout.addWidget(self.undo_btn, 0, 0)
        actions_layout.addWidget(self.redo_btn, 0, 1)
        actions_layout.addWidget(self.clear_char_btn, 0, 2)
        
        actions_layout.addWidget(self.vectorize_glyph_btn, 1, 0, 1, 3)
        
        actions_layout.addWidget(self.to_corner_btn, 2, 0)
        actions_layout.addWidget(self.to_smooth_btn, 2, 1)
        # [核心新增] 将新按钮添加到布局，跨越1列
        actions_layout.addWidget(self.to_asymmetric_btn, 2, 2)

        actions_layout.addWidget(self.close_path_btn, 3, 0)
        actions_layout.addWidget(self.break_path_btn, 3, 1, 1, 2)
        
        actions_layout.addWidget(self.insert_node_btn, 4, 0)
        actions_layout.addWidget(self.delete_node_btn, 4, 1, 1, 2)
        
        actions_layout.addWidget(self.merge_nodes_btn, 5, 0)
        
        actions_layout.addWidget(self.curve_to_line_btn, 6, 0)
        actions_layout.addWidget(self.line_to_curve_btn, 6, 1, 1, 2)
        
        layout.addWidget(actions_group)

        # --- 步骤 6: 添加拉伸因子并将最终的 widget 添加到父标签页 ---
        layout.addStretch()
        parent_tab_widget.addTab(widget, "工具")

        
    def _populate_layers_sub_tab(self, parent_tab_widget: QTabWidget):
        """[UI还原][已升级] 填充“设计”->“图层”子标签页的内容。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 10, 5, 10)
        
        # --- 笔画图层列表 ---
        layers_group = QGroupBox("笔画图层")
        layers_layout = QVBoxLayout(layers_group)
        
        self.layer_list_widget = QListWidget()
        self.layer_list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.layer_list_widget.setAlternatingRowColors(True)
        # 允许双击重命名
        self.layer_list_widget.setEditTriggers(QAbstractItemView.DoubleClicked)
        layers_layout.addWidget(self.layer_list_widget)
        
        # --- [核心新增] 透明度控制 ---
        opacity_group = QGroupBox("图层不透明度")
        opacity_layout = QHBoxLayout(opacity_group)
        self.layer_opacity_slider = QSlider(Qt.Horizontal)
        self.layer_opacity_slider.setRange(0, 100)
        self.layer_opacity_label = QLabel("100%")
        self.layer_opacity_label.setFixedWidth(40)
        opacity_layout.addWidget(self.layer_opacity_slider)
        opacity_layout.addWidget(self.layer_opacity_label)
        layers_layout.addWidget(opacity_group)
        
        # --- 图层操作按钮 ---
        btn_layout = QHBoxLayout()
        self.add_layer_btn = QPushButton("➕ 新建"); self.del_layer_btn = QPushButton("🗑️ 删除")
        self.up_layer_btn = QPushButton("🔼 上移"); self.down_layer_btn = QPushButton("🔽 下移")
        btn_layout.addWidget(self.add_layer_btn); btn_layout.addWidget(self.del_layer_btn)
        btn_layout.addWidget(self.up_layer_btn); btn_layout.addWidget(self.down_layer_btn)
        layers_layout.addLayout(btn_layout)
        
        layout.addWidget(layers_group)
        parent_tab_widget.addTab(widget, "图层")
        
    def _populate_components_sub_tab(self, parent_tab_widget: QTabWidget):
        """[UI还原] 填充“设计”->“部件”子标签页的内容。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(15)

        # 部件库卡片
        list_card = CardWidget("部件库", theme=Theme.LIGHT)
        list_card_layout = QVBoxLayout(list_card.contentWidget())
        self.component_list = QListWidget(); self.component_list.setAlternatingRowColors(True)
        list_card_layout.addWidget(self.component_list, 1)
        
        self.component_preview = QLabel("选择部件以预览")
        self.component_preview.setAlignment(Qt.AlignCenter); self.component_preview.setMinimumHeight(100)
        self.component_preview.setStyleSheet(f"background-color: {Theme.LIGHT['bg-tertiary']}; border: 1px solid {Theme.LIGHT['border-primary']}; border-radius: 4px; color: {Theme.LIGHT['content-secondary']};")
        list_card_layout.addWidget(self.component_preview)
        layout.addWidget(list_card, 1)

        # 操作卡片
        actions_card = CardWidget("部件操作", theme=Theme.LIGHT)
        actions_layout = QVBoxLayout(actions_card.contentWidget())
        self.save_as_component_btn = QPushButton("保存当前设计为新部件")
        self.insert_component_btn = QPushButton("将选中部件插入当前字符")
        self.delete_component_btn = QPushButton("删除选中部件")
        self.insert_component_btn.setEnabled(False); self.delete_component_btn.setEnabled(False)
        self.delete_component_btn.setStyleSheet(f"background-color: {Theme.LIGHT['danger']};")
        actions_layout.addWidget(self.save_as_component_btn)
        actions_layout.addWidget(self.insert_component_btn)
        actions_layout.addWidget(self.delete_component_btn)
        layout.addWidget(actions_card)
        
        parent_tab_widget.addTab(widget, "部件")
        
    def _populate_ai_sub_tab(self, parent_tab_widget: QTabWidget):
        """[UI还原] 填充“设计”->“AI助手”子标签页的内容。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(15)

        ai_card = CardWidget("AI 辅助生成", theme=Theme.LIGHT)
        card_layout = QVBoxLayout(ai_card.contentWidget())
        card_layout.setSpacing(10)

        card_layout.addWidget(QLabel("输入您的创意描述 (Prompt):"))
        self.ai_prompt_text = QTextEdit(); self.ai_prompt_text.setPlaceholderText("例如: 狂野的草书风格, 瘦金书风格...")
        self.ai_prompt_text.setFixedHeight(80)
        card_layout.addWidget(self.ai_prompt_text)

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("风格强度:"))
        self.ai_style_slider = QSlider(Qt.Horizontal); self.ai_style_slider.setRange(1, 100); self.ai_style_slider.setValue(70)
        params_layout.addWidget(self.ai_style_slider)
        self.ai_style_label = QLabel("0.7"); self.ai_style_label.setMinimumWidth(30)
        params_layout.addWidget(self.ai_style_label)
        card_layout.addLayout(params_layout)
        
        self.ai_generate_button = QPushButton("✨ 启动 AI 生成 ✨")
        card_layout.addWidget(self.ai_generate_button)
        
        self.ai_status_label = QLabel("AI 助手已就绪"); self.ai_status_label.setAlignment(Qt.AlignCenter)
        self.ai_status_label.setStyleSheet(f"color: {Theme.LIGHT['content-secondary']};")
        card_layout.addWidget(self.ai_status_label)
        
        layout.addWidget(ai_card)
        layout.addStretch()
        parent_tab_widget.addTab(widget, "AI助手")
    def _create_layers_tab(self, parent_tab_widget: QTabWidget):
        """
        创建并填充“图层”标签页。
        
        此方法使用 CardWidget 将图层列表和相关操作按钮组织在一起，
        提供了清晰的图层管理界面。

        [已修正] 
        - 解决了 CardWidget 缺少 theme 参数的 TypeError。
        - 确保在 CardWidget 中正确设置布局。
        - 图层列表现在倒序显示，符合设计习惯。
        """
        # --- 1. 创建该标签页的主容器和布局 ---
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # --- 2. 创建“图层管理”卡片 ---
        layers_card = CardWidget("图层管理", theme=Theme.LIGHT)
        
        # 获取卡片的内容区并为其设置布局
        card_layout = QVBoxLayout(layers_card.contentWidget())
        card_layout.setContentsMargins(0, 0, 0, 0) # 内容区内部无边距
        card_layout.setSpacing(10)

        # --- 3. 创建图层列表 ---
        self.layer_list_widget = QListWidget()
        self.layer_list_widget.setAlternatingRowColors(True) # 斑马条纹
        # 允许通过拖拽来重新排序图层
        self.layer_list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        # TODO: 需要连接 self.layer_list_widget.model().rowsMoved 信号来更新后端数据模型
        
        card_layout.addWidget(self.layer_list_widget, 1) # 列表占据大部分空间

        # --- 4. 创建图层操作按钮 ---
        btn_layout = QHBoxLayout()
        self.add_layer_btn = QPushButton("新建")
        self.del_layer_btn = QPushButton("删除")
        self.up_layer_btn = QPushButton("上移")
        self.down_layer_btn = QPushButton("下移")
        
        # 初始时禁用需要选中项的按钮
        self.del_layer_btn.setEnabled(False)
        self.up_layer_btn.setEnabled(False)
        self.down_layer_btn.setEnabled(False)
        
        # 将按钮添加到布局中
        btn_layout.addWidget(self.add_layer_btn)
        btn_layout.addWidget(self.del_layer_btn)
        btn_layout.addWidget(self.up_layer_btn)
        btn_layout.addWidget(self.down_layer_btn)
        card_layout.addLayout(btn_layout)

        # 将填充好的卡片添加到主布局，拉伸因子为1使其填充所有可用空间
        layout.addWidget(layers_card, 1)

        # --- 5. 将最终的 widget 添加到父标签页 ---
        parent_tab_widget.addTab(widget, "图层")

    def _create_preview_tab(self, parent_tab_widget: QTabWidget):
        """[UI还原] 创建并填充“预览”标签页，还原彩色卡片标题。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # --- 字符预览卡片 ---
        char_preview_card = CardWidget("字符预览", theme=Theme.LIGHT)
        char_preview_card.setTitleBarColor(Theme.LIGHT['accent-primary'], 'white')
        
        char_card_content_layout = QVBoxLayout(char_preview_card.contentWidget())
        char_card_content_layout.setContentsMargins(12, 12, 12, 12)
        
        self.preview_label = QLabel("选择一个字符以预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(200, 200)
        self.preview_label.setStyleSheet(f"background-color: {Theme.LIGHT['bg-tertiary']}; border: 1px solid {Theme.LIGHT['border-primary']}; border-radius: 4px;")
        
        char_card_content_layout.addWidget(self.preview_label)
        layout.addWidget(char_preview_card)

        # --- 文本预览卡片 ---
        text_preview_card = CardWidget("文本预览", theme=Theme.LIGHT)
        text_preview_card.setTitleBarColor(Theme.LIGHT['accent-primary'], 'white')
        
        text_card_content_layout = QVBoxLayout(text_preview_card.contentWidget())
        text_card_content_layout.setContentsMargins(12, 12, 12, 12)
        text_card_content_layout.setSpacing(10)
        
        text_card_content_layout.addWidget(QLabel("预览文本:"))
        self.preview_text_edit = QLineEdit("中国汉字书法艺术")
        text_card_content_layout.addWidget(self.preview_text_edit)
        
        # 实例化我们自定义的 TextPreviewWidget
        self.text_preview_widget = MainWindow.TextPreviewWidget(self)
        text_card_content_layout.addWidget(self.text_preview_widget)
        
        layout.addWidget(text_preview_card, 1)

        parent_tab_widget.addTab(widget, "👁️ 预览")

    def _create_analysis_tab(self, parent_tab_widget: QTabWidget):
        """
        [UI还原] 创建并填充“分析”标签页，还原彩色卡片标题。
        """
        # --- 1. 创建该标签页的主容器和布局 ---
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # --- 2. 创建“信息统计”卡片 ---
        stats_card = CardWidget("信息统计", theme=Theme.LIGHT)
        # 设置一个醒目的、与主题色一致的彩色标题栏
        stats_card.setTitleBarColor(Theme.LIGHT['accent-primary'], 'white')
        
        stats_card_layout = QVBoxLayout(stats_card.contentWidget())
        stats_card_layout.setContentsMargins(12, 12, 12, 12)
        
        # --- 3. 创建用于显示富文本的 QTextEdit ---
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True) # 设置为只读，用户不能编辑
        # 设置样式，使其看起来更像一个信息面板而不是输入框
        self.stats_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.LIGHT['bg-tertiary']};
                border: none;
                color: {Theme.LIGHT['content-secondary']};
                font-size: 10pt;
                line-height: 1.5;
            }}
        """)
        
        stats_card_layout.addWidget(self.stats_text)
        layout.addWidget(stats_card, 1) # 让卡片填充所有可用空间
        
        # 添加一个伸缩项，确保卡片不会被拉伸得过大（虽然这里只有一个控件）
        layout.addStretch()
        
        # 将最终完成的 widget 添加到父标签页
        parent_tab_widget.addTab(widget, "📊 分析")
    def _create_settings_tab(self, parent_tab_widget: QTabWidget):
        """[已完善] 创建并填充“设置”标签页，用于编辑字体全局属性。"""
        # --- 1. 创建主滚动区域，以防内容过多 ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        widget = QWidget()
        scroll_area.setWidget(widget)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # --- 2. 字体基础信息卡片 ---
        info_card = CardWidget("字体基础信息", theme=Theme.LIGHT)
        info_layout = QGridLayout(info_card.contentWidget())
        
        self.setting_family_name = QLineEdit(self.font_settings['familyName'])
        self.setting_style_name = QLineEdit(self.font_settings['styleName'])
        self.setting_version = QLineEdit(self.font_settings['version'])
        self.setting_copyright = QLineEdit(self.font_settings['copyright'])

        info_layout.addWidget(QLabel("字体家族名:"), 0, 0)
        info_layout.addWidget(self.setting_family_name, 0, 1)
        info_layout.addWidget(QLabel("样式名:"), 1, 0)
        info_layout.addWidget(self.setting_style_name, 1, 1)
        info_layout.addWidget(QLabel("版本号:"), 2, 0)
        info_layout.addWidget(self.setting_version, 2, 1)
        info_layout.addWidget(QLabel("版权信息:"), 3, 0)
        info_layout.addWidget(self.setting_copyright, 3, 1)
        
        layout.addWidget(info_card)

        # --- 3. 核心字体度量 (Font Metrics) 卡片 ---
        metrics_card = CardWidget("核心字体度量 (Font Metrics)", theme=Theme.LIGHT)
        metrics_layout = QGridLayout(metrics_card.contentWidget())
        
        # a. Units Per Em (UPM)
        self.setting_upm_combo = QComboBox()
        self.setting_upm_combo.addItems(["1000", "1024", "2048"])
        self.setting_upm_combo.setCurrentText(str(self.font_settings['unitsPerEm']))
        metrics_layout.addWidget(QLabel("UPM (网格大小):"), 0, 0)
        metrics_layout.addWidget(self.setting_upm_combo, 0, 1)

        # 辅助函数，用于创建滑块+标签组合
        def create_metric_slider(name, value, min_val, max_val):
            label = QLabel(f"{value}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(value)
            slider.valueChanged.connect(lambda v, l=label: l.setText(str(v)))
            return slider, label

        # b. Ascender (上伸部)
        self.setting_ascender_slider, self.setting_ascender_label = create_metric_slider(
            "上伸部:", self.font_settings['ascender'], 0, 1024)
        metrics_layout.addWidget(QLabel("上伸部 (Ascender):"), 1, 0)
        metrics_layout.addWidget(self.setting_ascender_slider, 1, 1)
        metrics_layout.addWidget(self.setting_ascender_label, 1, 2)

        # c. Descender (下伸部)
        self.setting_descender_slider, self.setting_descender_label = create_metric_slider(
            "下伸部:", self.font_settings['descender'], -512, 0)
        metrics_layout.addWidget(QLabel("下伸部 (Descender):"), 2, 0)
        metrics_layout.addWidget(self.setting_descender_slider, 2, 1)
        metrics_layout.addWidget(self.setting_descender_label, 2, 2)

        # d. x-Height (x 高度)
        self.setting_xheight_slider, self.setting_xheight_label = create_metric_slider(
            "x 高度:", self.font_settings['xHeight'], 0, 1024)
        metrics_layout.addWidget(QLabel("x 高度 (x-Height):"), 3, 0)
        metrics_layout.addWidget(self.setting_xheight_slider, 3, 1)
        metrics_layout.addWidget(self.setting_xheight_label, 3, 2)

        # e. Cap Height (大写高度)
        self.setting_capheight_slider, self.setting_capheight_label = create_metric_slider(
            "大写高度:", self.font_settings['capHeight'], 0, 1024)
        metrics_layout.addWidget(QLabel("大写高度 (Cap Height):"), 4, 0)
        metrics_layout.addWidget(self.setting_capheight_slider, 4, 1)
        metrics_layout.addWidget(self.setting_capheight_label, 4, 2)
        
        layout.addWidget(metrics_card)

        layout.addStretch()
        parent_tab_widget.addTab(scroll_area, "⚙️ 设置")

    def _create_export_tab(self, parent_tab_widget: QTabWidget):
        """[已完善] 创建并填充“导出”标签页。"""
        # --- 1. 创建主滚动区域 ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        widget = QWidget()
        scroll_area.setWidget(widget)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # --- 2. TTF字体导出卡片 ---
        ttf_card = CardWidget("TTF字体导出", theme=Theme.LIGHT)
        ttf_layout = QVBoxLayout(ttf_card.contentWidget())
        
        # a. 检查 fontTools 是否可用
        if not FONTTOOLS_AVAILABLE:
            error_label = QLabel("⚠️ `fonttools` 库未安装。\nTTF导出功能不可用。\n请运行: pip install fonttools")
            error_label.setStyleSheet(f"color: {Theme.LIGHT['danger']};")
            ttf_layout.addWidget(error_label)
        else:
            # b. 导出选项
            options_layout = QGridLayout()
            self.export_subset_checkbox = QCheckBox("字体子集化 (仅包含已设计的字符)")
            self.export_subset_checkbox.setChecked(True)
            self.export_subset_checkbox.setToolTip("勾选后，生成的字体文件将只包含您设计过的字符，文件体积更小。")
            
            options_layout.addWidget(self.export_subset_checkbox, 0, 0, 1, 2)
            ttf_layout.addLayout(options_layout)

            # c. 导出按钮
            self.export_ttf_button = QPushButton("导出为 .TTF 文件...")
            self.export_ttf_button.setStyleSheet(f"background-color: {Theme.LIGHT['success']};")
            ttf_layout.addWidget(self.export_ttf_button)

        layout.addWidget(ttf_card)
        
        # --- 3. 图像导出 (当前字符) 卡片 ---
        image_card = CardWidget("图像导出 (当前字符)", theme=Theme.LIGHT)
        image_layout = QGridLayout(image_card.contentWidget())

        # a. 尺寸设置
        self.image_export_size_slider, self.image_export_size_label = self._create_slider_with_label(512, 64, 4096)
        image_layout.addWidget(QLabel("图像尺寸 (px):"), 0, 0)
        image_layout.addWidget(self.image_export_size_slider, 0, 1)
        image_layout.addWidget(self.image_export_size_label, 0, 2)
        
        # b. 选项
        self.image_export_transparent_bg = QCheckBox("透明背景")
        self.image_export_transparent_bg.setChecked(True)
        image_layout.addWidget(self.image_export_transparent_bg, 1, 0, 1, 3)

        # c. 导出按钮
        buttons_layout = QHBoxLayout()
        self.export_png_button = QPushButton("导出为 .PNG")
        self.export_svg_button = QPushButton("导出为 .SVG")
        buttons_layout.addWidget(self.export_png_button)
        buttons_layout.addWidget(self.export_svg_button)
        image_layout.addLayout(buttons_layout, 2, 0, 1, 3)

        layout.addWidget(image_card)

        layout.addStretch()
        parent_tab_widget.addTab(scroll_area, "📤 导出")

    def _create_slider_with_label(self, value: int, min_val: int, max_val: int) -> Tuple[QSlider, QLabel]:
        """[新增辅助] 创建一个滑块和与其关联的实时显示数值的标签。"""
        label = QLabel(str(value))
        label.setMinimumWidth(35)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(value)
        slider.valueChanged.connect(lambda v, l=label: l.setText(str(v)))
        return slider, label
    
    def _create_history_tab(self, parent_tab_widget: QTabWidget):
        """[已完善] 创建并填充“历史”标签页，用于显示撤销/重做栈。"""
        # --- 1. 创建主容器和布局 ---
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(15)

        # --- 2. 创建“操作历史记录”卡片 ---
        history_card = CardWidget("操作历史记录", theme=Theme.LIGHT)
        card_layout = QVBoxLayout(history_card.contentWidget())
        
        # --- 3. [核心] 创建 QUndoView 并关联 QUndoStack ---
        # QUndoView 是专门用来可视化 QUndoStack 的Qt标准控件
        self.history_view = QUndoView(self.undo_stack)
        self.history_view.setStyleSheet(f"""
            QUndoView {{
                background-color: {Theme.LIGHT['bg-tertiary']};
                border: none;
            }}
        """)
        # 设置当没有操作时显示的提示文字
        self.history_view.setEmptyLabel("尚无任何操作")
        
        card_layout.addWidget(self.history_view)
        
        # --- 4. 添加“清空历史”按钮 ---
        self.clear_history_button = QPushButton("清空历史记录")
        self.clear_history_button.setStyleSheet(f"background-color: {Theme.LIGHT['danger']};")
        card_layout.addWidget(self.clear_history_button)
        
        layout.addWidget(history_card, 1) # 让卡片填充所有可用空间
        
        parent_tab_widget.addTab(widget, "📜 历史")

    def _create_kerning_tab(self, parent_tab_widget: QTabWidget):
        """[已完善] 创建并填充“字跨”标签页。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 15, 10, 10); layout.setSpacing(15)

        # --- 1. 实时预览卡片 ---
        preview_card = CardWidget("实时预览", theme=Theme.LIGHT)
        preview_layout = QVBoxLayout(preview_card.contentWidget())
        self.kerning_preview_widget = MainWindow.KerningPreviewWidget(self)
        preview_layout.addWidget(self.kerning_preview_widget)
        layout.addWidget(preview_card)
        
        # --- 2. 调整字符对卡片 ---
        control_card = CardWidget("调整字符对", theme=Theme.LIGHT)
        control_layout = QGridLayout(control_card.contentWidget())
        
        self.kerning_pair_edit = QLineEdit("你我")
        self.kerning_pair_edit.setMaxLength(20) # 限制一下长度
        self.kerning_pair_edit.setPlaceholderText("输入要调整的字符对...")
        
        self.kerning_value_slider, self.kerning_value_label = self._create_slider_with_label(0, -200, 200)
        
        control_layout.addWidget(QLabel("字符对:"), 0, 0)
        control_layout.addWidget(self.kerning_pair_edit, 0, 1, 1, 2)
        control_layout.addWidget(QLabel("调整值:"), 1, 0)
        control_layout.addWidget(self.kerning_value_slider, 1, 1)
        control_layout.addWidget(self.kerning_value_label, 1, 2)
        layout.addWidget(control_card)

        # --- 3. 已保存的字偶距对卡片 ---
        saved_card = CardWidget("已保存的字偶距对", theme=Theme.LIGHT)
        saved_layout = QVBoxLayout(saved_card.contentWidget())
        
        self.kerning_list_widget = QListWidget()
        self.kerning_list_widget.setAlternatingRowColors(True)
        saved_layout.addWidget(self.kerning_list_widget, 1)
        
        btn_layout = QHBoxLayout()
        self.save_kerning_btn = QPushButton("保存/更新当前对")
        self.delete_kerning_btn = QPushButton("删除选中对")
        self.delete_kerning_btn.setStyleSheet(f"background-color: {Theme.LIGHT['danger']};")
        self.delete_kerning_btn.setEnabled(False)
        btn_layout.addWidget(self.save_kerning_btn)
        btn_layout.addWidget(self.delete_kerning_btn)
        saved_layout.addLayout(btn_layout)

        layout.addWidget(saved_card, 1)
        parent_tab_widget.addTab(widget, "↔️ 字跨")

    def _update_kerning_preview(self):
        """[新增辅助] 根据当前UI控件的值更新字偶距实时预览。"""
        if hasattr(self, 'kerning_preview_widget'):
            pair_text = self.kerning_pair_edit.text()
            value = self.kerning_value_slider.value()
            self.kerning_preview_widget.set_pair(pair_text, value)

    def _update_kerning_list(self):
        """[新增辅助] 从 self.kerning_pairs 字典刷新已保存的字偶距对列表。"""
        self.kerning_list_widget.clear()
        for pair, value in sorted(self.kerning_pairs.items()):
            item = QListWidgetItem(f"{pair} → {value}")
            item.setData(Qt.UserRole, pair) # 存储键以便查找
            self.kerning_list_widget.addItem(item)
        self.delete_kerning_btn.setEnabled(False)

    def on_kerning_list_selection_changed(self):
        """[槽][新增] 当用户在已保存列表中选择一项时，加载其数据到编辑控件。"""
        selected_items = self.kerning_list_widget.selectedItems()
        self.delete_kerning_btn.setEnabled(bool(selected_items))
        
        if selected_items:
            item = selected_items[0]
            pair = item.data(Qt.UserRole)
            value = self.kerning_pairs.get(pair, 0)
            
            # 暂时断开信号以避免循环更新
            self.kerning_pair_edit.blockSignals(True)
            self.kerning_value_slider.blockSignals(True)
            
            self.kerning_pair_edit.setText(pair)
            self.kerning_value_slider.setValue(value)
            
            self.kerning_pair_edit.blockSignals(False)
            self.kerning_value_slider.blockSignals(False)
            
            self._update_kerning_preview()

    def on_save_kerning_pair(self):
        """[槽][新增] 保存或更新当前编辑的字偶距对。"""
        pair_text = self.kerning_pair_edit.text()
        if len(pair_text) < 2:
            QMessageBox.warning(self, "输入无效", "请输入至少两个字符来定义一个字偶距对。")
            return
        
        pair_key = pair_text[:2]
        value = self.kerning_value_slider.value()
        
        self.kerning_pairs[pair_key] = value
        self.is_project_dirty = True
        self._update_kerning_list()
        self.statusBar().showMessage(f"已保存字偶距对 '{pair_key}' → {value}", 3000)

    def on_delete_kerning_pair(self):
        """[槽][新增] 删除选中的字偶距对。"""
        selected_items = self.kerning_list_widget.selectedItems()
        if not selected_items:
            return
            
        pair_key = selected_items[0].data(Qt.UserRole)
        reply = QMessageBox.question(self, "确认删除", f"您确定要删除字偶距对 '{pair_key}' 吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if pair_key in self.kerning_pairs:
                del self.kerning_pairs[pair_key]
                self.is_project_dirty = True
                self._update_kerning_list()
                self.statusBar().showMessage(f"已删除字偶距对 '{pair_key}'", 3000)
    def _connect_signals(self):
        """
        [控制器][V5.6 设置面板增强版] 集中连接所有UI组件的信号到本类的槽函数。

        此版本新增了对“设置”面板中所有控件的信号连接，实现了UI与
        `self.font_settings` 数据模型的双向绑定。
        """
        # --- 1. 导航栏和菜单 Actions 信号 ---
        self.new_action.triggered.connect(self.on_new_project)
        self.load_db_action.triggered.connect(self.on_load_database)
        self.open_action.triggered.connect(self.on_open_project)
        self.save_action.triggered.connect(self.on_save_project)
        self.save_as_action.triggered.connect(self.on_save_project_as)
        self.export_ttf_action.triggered.connect(self.on_export_ttf)
        self.exit_action.triggered.connect(self.close)

        # --- 2. 中央画布 (DrawingCanvas) 自定义信号 ---
        self.canvas.stroke_finished.connect(self.on_stroke_finished)
        self.canvas.stroke_modified.connect(self.on_stroke_modified)

        # --- 3. 字符浏览器信号 ---
        self.char_list_view.clicked.connect(self.on_char_selected)
        self.search_box.textChanged.connect(self._perform_search)
        self.category_combo.currentIndexChanged.connect(self._perform_search)
        self.char_as_ref_checkbox.stateChanged.connect(self.on_toggle_reference_image)
        self.designed_checkbox.stateChanged.connect(self._perform_search)
        self.undesigned_checkbox.stateChanged.connect(self._perform_search)

        # --- 4. 工具面板信号 ---
        # 4.1 工具选择按钮
        self.tool_button_group.buttonClicked.connect(self._on_tool_selected)
        
        # 4.2 笔画参数
        self.stroke_width_slider.valueChanged.connect(self._on_stroke_width_changed)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        self.color_btn.clicked.connect(self._on_color_btn_clicked)
        self.pressure_checkbox.stateChanged.connect(self._on_pressure_changed)
        self.antialias_checkbox.stateChanged.connect(self._on_antialias_changed)
        
        # 4.3 显示选项
        self.show_grid_checkbox.stateChanged.connect(self._on_grid_options_changed)
        self.show_guides_checkbox.stateChanged.connect(self._on_grid_options_changed)
        self.show_metrics_checkbox.stateChanged.connect(self._on_grid_options_changed)
        self.show_pro_guides_checkbox.stateChanged.connect(self._on_grid_options_changed)
        
        # 4.4 编辑操作
        self.clear_char_btn.clicked.connect(self.on_clear_char)
        self.undo_btn.clicked.connect(self.undo_action.trigger)
        self.redo_btn.clicked.connect(self.redo_action.trigger)
        self.vectorize_glyph_btn.clicked.connect(self.on_vectorize_glyph)
        self.to_corner_btn.clicked.connect(self.on_convert_to_corner)
        self.to_smooth_btn.clicked.connect(self.on_convert_to_smooth)
        self.to_asymmetric_btn.clicked.connect(self.on_convert_to_asymmetric)
        self.close_path_btn.clicked.connect(self.on_close_path)
        self.break_path_btn.clicked.connect(self.on_break_path)
        self.insert_node_btn.clicked.connect(self.on_toggle_insert_node_mode)
        self.delete_node_btn.clicked.connect(self.on_delete_node)
        self.merge_nodes_btn.clicked.connect(self.on_merge_nodes)
        self.curve_to_line_btn.clicked.connect(self.on_convert_segment_to_line)
        self.line_to_curve_btn.clicked.connect(self.on_convert_segment_to_curve)

        # --- 5. 图层面板信号 ---
        self.add_layer_btn.clicked.connect(self.on_add_layer)
        self.del_layer_btn.clicked.connect(self.on_remove_layer)
        self.up_layer_btn.clicked.connect(self.on_move_layer_up)
        self.down_layer_btn.clicked.connect(self.on_move_layer_down)
        self.layer_list_widget.itemSelectionChanged.connect(self._update_layer_buttons_state)
        self.layer_list_widget.itemDoubleClicked.connect(self.on_layer_rename_start)
        self.layer_opacity_slider.sliderReleased.connect(self.on_layer_opacity_set)
        self.layer_opacity_slider.valueChanged.connect(lambda value: self.layer_opacity_label.setText(f"{value}%"))
        
        # --- 6. 预览面板信号 ---
        self.preview_text_edit.textChanged.connect(self.update_previews)

        # --- 7. 部件面板信号 ---
        self.component_list.currentItemChanged.connect(self.on_component_selected)
        self.save_as_component_btn.clicked.connect(self.on_save_as_component)
        self.insert_component_btn.clicked.connect(self.on_insert_component)
        self.delete_component_btn.clicked.connect(self.on_delete_component)

        # --- 8. AI 助手面板信号 ---
        self.ai_style_slider.valueChanged.connect(self._on_ai_style_changed)
        self.ai_generate_button.clicked.connect(self.on_ai_generate)
        
        # --- 9. [核心新增] 属性面板信号 ---
        self.segment_tension_slider.sliderReleased.connect(self.on_segment_tension_changed)
        self.segment_tension_slider.valueChanged.connect(
            lambda val: self.segment_tension_label.setText(f"{val}%")
        )

        # --- 10. [核心新增] 设置面板信号 ---
        # 使用 lambda 函数将控件的变动直接更新到 self.font_settings 字典中
        self.setting_family_name.textChanged.connect(lambda t: self._update_font_setting('familyName', t))
        self.setting_style_name.textChanged.connect(lambda t: self._update_font_setting('styleName', t))
        self.setting_version.textChanged.connect(lambda t: self._update_font_setting('version', t))
        self.setting_copyright.textChanged.connect(lambda t: self._update_font_setting('copyright', t))
        self.setting_upm_combo.currentTextChanged.connect(lambda t: self._update_font_setting('unitsPerEm', int(t)))
        self.setting_ascender_slider.valueChanged.connect(lambda v: self._update_font_setting('ascender', v))
        self.setting_descender_slider.valueChanged.connect(lambda v: self._update_font_setting('descender', v))
        self.setting_xheight_slider.valueChanged.connect(lambda v: self._update_font_setting('xHeight', v))
        self.setting_capheight_slider.valueChanged.connect(lambda v: self._update_font_setting('capHeight', v))

        # --- [核心新增] 11. 导出面板信号 ---
        if FONTTOOLS_AVAILABLE:
            self.export_ttf_button.clicked.connect(self.export_ttf_action.trigger)
        
        self.export_png_button.clicked.connect(self.on_export_png)
        self.export_svg_button.clicked.connect(self.on_export_svg)

        # --- [核心新增] 12. 历史面板信号 ---
        self.clear_history_button.clicked.connect(self.undo_stack.clear)

        # --- [核心新增] 13. 字跨面板信号 ---
        self.kerning_pair_edit.textChanged.connect(self._update_kerning_preview)
        self.kerning_value_slider.valueChanged.connect(self._update_kerning_preview)
        self.save_kerning_btn.clicked.connect(self.on_save_kerning_pair)
        self.delete_kerning_btn.clicked.connect(self.on_delete_kerning_pair)
        self.kerning_list_widget.itemSelectionChanged.connect(self.on_kerning_list_selection_changed)
    
    def on_export_png(self):
        """[槽][新增] 导出当前字符为 PNG 图像。"""
        if not self.current_char_obj:
            QMessageBox.warning(self, "操作无效", "请先选择一个要导出的字符。")
            return
            
        char = self.current_char_obj.char
        filename, _ = QFileDialog.getSaveFileName(self, f"导出 '{char}' 为 PNG", f"{char}.png", "PNG 图像 (*.png)")
        
        if not filename:
            return

        size = self.image_export_size_slider.value()
        bg_color = Qt.transparent if self.image_export_transparent_bg.isChecked() else Qt.white
        
        try:
            pixmap = self.current_char_obj.get_preview_image(
                all_components=self.components,
                size=size,
                bg_color=bg_color
            )
            if pixmap.save(filename, "PNG"):
                self.statusBar().showMessage(f"成功导出 PNG 到: {os.path.basename(filename)}", 5000)
            else:
                raise IOError("保存Pixmap失败。")
        except Exception as e:
            QMessageBox.critical(self, "PNG 导出失败", f"导出图像时发生错误: {e}")

    def on_export_svg(self):
        """[槽][新增] 导出当前字符为 SVG 矢量图像。"""
        if not self.current_char_obj:
            QMessageBox.warning(self, "操作无效", "请先选择一个要导出的字符。")
            return

        char = self.current_char_obj.char
        filename, _ = QFileDialog.getSaveFileName(self, f"导出 '{char}' 为 SVG", f"{char}.svg", "SVG 图像 (*.svg)")

        if not filename:
            return
        
        try:
            size = self.image_export_size_slider.value()
            path = self.current_char_obj.to_vector_path(self.components)
            bounds = path.get_bounds()

            if not bounds:
                raise ValueError("无法导出空字符。")

            min_x, min_y, max_x, max_y = bounds
            content_w = max_x - min_x
            content_h = max_y - min_y
            
            # 计算缩放和平移以适应SVG视图框
            padding = size * 0.1
            view_size = size
            
            scale = (view_size - 2 * padding) / max(content_w, content_h, 1.0)
            tx = (view_size - content_w * scale) / 2
            ty = (view_size - content_h * scale) / 2
            
            svg_content = f'<svg width="{view_size}" height="{view_size}" xmlns="http://www.w3.org/2000/svg">\n'
            
            # 如果背景不透明，则添加一个白色矩形背景
            if not self.image_export_transparent_bg.isChecked():
                svg_content += f'  <rect width="100%" height="100%" fill="white"/>\n'
            
            # 添加路径数据，应用变换
            svg_content += (f'  <path d="{path.to_svg_path_data()}" '
                            f'transform="translate({tx}, {ty}) scale({scale}) translate({-min_x}, {-min_y})" '
                            'fill="black" stroke="none"/>\n')
            
            svg_content += '</svg>'

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            self.statusBar().showMessage(f"成功导出 SVG 到: {os.path.basename(filename)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "SVG 导出失败", f"导出图像时发生错误: {e}")
    def _update_font_setting(self, key: str, value: Any):
        """[新增辅助] 更新字体设置字典中的一个值，并标记项目为已修改。"""
        if self.font_settings.get(key) != value:
            self.font_settings[key] = value
            self.is_project_dirty = True
            # 特殊处理：如果UPM变化，需要更新全局的 grid_size
            if key == 'unitsPerEm':
                self.grid_size = value
                if self.canvas:
                    self.canvas.grid_size = value
                    self.canvas.update() # 强制画布重绘以适应新网格
            # 如果是度量线变化，也需要重绘画布
            if key in ['ascender', 'descender', 'xHeight', 'capHeight']:
                 if self.canvas:
                    self.canvas.update() # 强制画布重绘以更新度量线
    # 添加新方法
    def update_node_tool_buttons(self):
        """根据当前画布的选择状态，更新节点工具按钮的可用性。"""
        self._update_node_action_buttons()


    # 添加新槽函数

    def on_convert_to_asymmetric(self):
        """[槽][新增] 连接到“转为非对称”按钮。"""
        self.on_set_anchor_type('asymmetric')
    def on_set_anchor_type(self, new_type: Literal['corner', 'smooth', 'asymmetric']):
        """
        [槽][已增强] 将所有选中的锚点设置为指定类型，并使用 QUndoCommand。
        现在支持 'corner', 'smooth', 和 'asymmetric' 三种类型。
        """
        canvas = self.canvas
        if not (canvas and canvas.current_char and 
                canvas.selected_stroke_index != -1 and canvas.selected_anchor_indices):
            return

        stroke_index = canvas.selected_stroke_index
        stroke = canvas.current_char.strokes[stroke_index]
        indices_to_change = canvas.selected_anchor_indices

        # 准备新的 anchor_types 列表
        new_anchor_types = stroke.anchor_types[:]
        changed = False
        for idx in indices_to_change:
            if 0 <= idx < len(new_anchor_types):
                if new_anchor_types[idx] != new_type:
                    new_anchor_types[idx] = new_type
                    changed = True
        
        if changed:
            description = f"将 {len(indices_to_change)} 个节点转为 {new_type}"
            command = ModifyStrokePropertyCommand(
                self, stroke_index, 'anchor_types', new_anchor_types, description
            )
            
            # [关键] 属性改变后，必须让矢量路径缓存失效，以便重新计算几何形状
            # 注意: 这个回调函数在之前的增强中已经添加，这里只是确保它被正确使用
            command.set_invalidate_callback(lambda s: s._invalidate_vectorization_cache())
            self.undo_stack.push(command)


    # 添加新方法 (用于连接信号)
    def on_convert_to_corner(self):
        """[槽] 连接到“转为尖角”按钮。"""
        self.on_set_anchor_type('corner')
    def on_convert_to_smooth(self):
        """[槽] 连接到“转为平滑”按钮。"""
        self.on_set_anchor_type('smooth')
    def on_close_path(self):
        """[槽] 连接到“闭合路径”按钮。"""
        self.canvas.close_selected_path()
        
    def on_break_path(self):
        """[槽] 连接到“断开路径”按钮。"""
        self.canvas.break_path_at_selected_node()

    def on_delete_node(self):
        """[槽] 连接到“删除节点”按钮。"""
        self.canvas.delete_selected_node()
    def on_toggle_insert_node_mode(self, checked: bool):
        """[槽] 切换插入节点模式。"""
        if not self.canvas: return
        
        self.canvas.is_insert_mode = checked
        
        if checked:
            if self.current_tool != 'node_editor':
                self.tool_buttons['node_editor'].click()
            self.canvas.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage("插入节点模式：请在选中的笔画路径上点击以添加新节点。", 5000)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("已退出插入节点模式。")

    def on_insert_node_at_pos(self, canvas_pos: QPoint):
        """[槽] 由画布调用，在指定位置执行节点插入，并确保正确退出插入模式。"""
        if not (self.canvas and self.canvas.current_char and self.canvas.selected_stroke_index != -1):
            self.insert_node_btn.setChecked(False)
            return

        stroke_index = self.canvas.selected_stroke_index
        old_stroke = self.canvas.current_char.strokes[stroke_index]
        path = old_stroke.to_bezier_path()
        grid_pos = self.canvas._to_grid_coords(canvas_pos)

        hit_info = self._find_closest_point_on_path(path, grid_pos)
        if hit_info:
            segment_index = hit_info['segment_index']
            t = hit_info['t']
            
            new_path = self._split_bezier_segment(path, segment_index, t)
            
            if new_path is not path:
                new_stroke = self.canvas._rebuild_stroke_from_path(old_stroke, new_path)
                cmd = ModifyStrokeCommand(self, stroke_index, old_stroke, new_stroke)
                self.undo_stack.push(cmd)
        
        self.insert_node_btn.blockSignals(True)
        self.insert_node_btn.setChecked(False)
        self.insert_node_btn.blockSignals(False)

        if self.canvas:
            self.canvas.is_insert_mode = False
            self.canvas.setCursor(Qt.ArrowCursor)

        self.statusBar().showMessage("已退出插入节点模式。", 3000)

    def _find_closest_point_on_path(self, path: VectorPath, grid_pos: Point, num_samples: int = 100) -> Optional[Dict]:
        """[辅助][最终修复版] 找到路径上离给定点最近的点，并返回详细信息。"""
        if not path.commands:
            return None

        min_dist_sq = float('inf')
        closest_info = None
        
        # [核心修复] 正确处理子路径和起始点
        subpath_start_p = None
        current_p = None

        for i, cmd in enumerate(path.commands):
            cmd_name = cmd[0]
            
            # 记录子路径的起点
            if cmd_name == 'moveTo':
                current_p = cmd[1]
                subpath_start_p = cmd[1]
                continue
            
            # 如果没有有效的起点，则跳过
            if current_p is None:
                continue

            p_start = current_p
            p_end = None
            
            # --- a. 处理直线段 ---
            if cmd_name == 'lineTo':
                p_end = cmd[1]
                # 矢量化计算点到线段的距离
                dist_sq = _perpendicular_distance_squared_to_segment_vectorized(
                    np.array([grid_pos]), np.array(p_start), np.array(p_end)
                )[0]
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    # 对于直线，我们简化为在中点插入
                    closest_info = {'segment_index': i, 't': 0.5, 'pos': p_start} # pos 仅为占位

            # --- b. 处理三次曲线段 ---
            elif cmd_name == 'curveTo':
                p_end = cmd[3]
                p0, c1, c2, p1 = p_start, cmd[1], cmd[2], cmd[3]
                
                # 在曲线上采样，找到最近的点和对应的 t 值
                for j in range(num_samples + 1):
                    t = j / num_samples
                    omt = 1 - t
                    x = omt**3 * p0[0] + 3 * omt**2 * t * c1[0] + 3 * omt * t**2 * c2[0] + t**3 * p1[0]
                    y = omt**3 * p0[1] + 3 * omt**2 * t * c1[1] + 3 * omt * t**2 * c2[1] + t**3 * p1[1]
                    
                    dist_sq = (x - grid_pos[0])**2 + (y - grid_pos[1])**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_info = {'segment_index': i, 't': t, 'pos': (x, y)}

            # --- c. 处理闭合路径的隐式线段 ---
            elif cmd_name == 'closePath' and subpath_start_p:
                p_end = subpath_start_p
                dist_sq = _perpendicular_distance_squared_to_segment_vectorized(
                    np.array([grid_pos]), np.array(p_start), np.array(p_end)
                )[0]
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_info = {'segment_index': i, 't': 0.5, 'pos': p_start}

            # 更新当前点为本段的终点
            current_p = p_end

        # 增加一个阈值，防止点击离路径太远也触发插入
        if min_dist_sq > (20 / self.canvas._pixel_size)**2: # 20像素的点击容差
             return None

        return closest_info

    def _split_bezier_segment(self, path: VectorPath, segment_index: int, t: float) -> VectorPath:
        """[核心算法][最终修复版] 使用 de Casteljau 算法在 t 处分裂贝塞尔曲线段。"""
        
        commands = list(path.commands)
        if not (0 < segment_index < len(commands)):
            return path # 索引无效

        cmd_to_split = commands[segment_index]
        
        # [核心修复] 同时支持分裂直线和曲线
        # --- a. 分裂直线段 ---
        if cmd_to_split[0] == 'lineTo':
            p1 = np.array(cmd_to_split[1])
            # 找到前一个锚点
            p0 = np.array(self._get_anchor_point_before(commands, segment_index))
            if p0 is None: return path
            
            # 计算中点
            mid_point = tuple(p0 + (p1 - p0) * 0.5)
            
            # 分裂成两条线段
            new_cmd1 = ('lineTo', mid_point)
            new_cmd2 = ('lineTo', tuple(p1))
            
            commands[segment_index] = new_cmd1
            commands.insert(segment_index + 1, new_cmd2)
            
            return VectorPath(commands)

        # --- b. 分裂曲线段 ---
        elif cmd_to_split[0] == 'curveTo':
            p1 = np.array(cmd_to_split[3])
            c1, c2 = np.array(cmd_to_split[1]), np.array(cmd_to_split[2])
            
            p0 = np.array(self._get_anchor_point_before(commands, segment_index))
            if p0 is None: return path

            # De Casteljau's algorithm
            p01 = (1 - t) * p0 + t * c1
            p12 = (1 - t) * c1 + t * c2
            p23 = (1 - t) * c2 + t * p1
            p012 = (1 - t) * p01 + t * p12
            p123 = (1 - t) * p12 + t * p23
            new_anchor = (1 - t) * p012 + t * p123

            # 创建新的命令
            new_cmd1 = ('curveTo', tuple(p01), tuple(p012), tuple(new_anchor))
            new_cmd2 = ('curveTo', tuple(p123), tuple(p23), tuple(p1))

            # 替换旧命令并插入新命令
            commands[segment_index] = new_cmd1
            commands.insert(segment_index + 1, new_cmd2)
            
            return VectorPath(commands)

        return path # 如果不是可分裂的段，返回原路径

    def _get_anchor_point_before(self, commands: list, index: int) -> Optional[Point]:
        """[新增辅助] 从命令列表中找到指定索引之前的最后一个锚点坐标。"""
        for i in range(index - 1, -1, -1):
            cmd = commands[i]
            if cmd[0] == 'moveTo' or cmd[0] == 'lineTo':
                return cmd[1]
            if cmd[0] == 'curveTo':
                return cmd[3]
        return None
    def on_merge_nodes(self):
        """[槽][已实现] 将多个选中的节点合并为一个新节点。"""
        canvas = self.canvas
        if not (canvas and canvas.current_char and 
                canvas.selected_stroke_index != -1 and len(canvas.selected_anchor_indices) > 1):
            return

        stroke_index = canvas.selected_stroke_index
        old_stroke = canvas.current_char.strokes[stroke_index]
        path = old_stroke.to_bezier_path()
        indices_to_merge = sorted(list(canvas.selected_anchor_indices), reverse=True)

        # 1. 计算合并后新节点的位置 (几何中心)
        points_to_merge = []
        for idx in indices_to_merge:
            pos = canvas._get_node_pos_from_path(path, {'type': 'anchor', 'anchor_index': idx})
            if pos:
                points_to_merge.append(pos)
        
        if not points_to_merge:
            return
            
        center_point = tuple(np.mean(np.array(points_to_merge), axis=0))

        # 2. 从路径命令中删除被合并的节点，并在第一个被合并节点的位置插入新节点
        new_commands = []
        anchor_idx_counter = -1
        first_merged_node_idx = indices_to_merge[-1] # 最小的索引

        for cmd in path.commands:
            is_anchor = cmd[0] in ['moveTo', 'lineTo', 'curveTo']
            if is_anchor:
                anchor_idx_counter += 1
            
            # 如果是第一个被合并的节点，用新节点替换它
            if is_anchor and anchor_idx_counter == first_merged_node_idx:
                if cmd[0] == 'moveTo':
                    new_commands.append(('moveTo', center_point))
                else: # lineTo or curveTo
                    # 简化处理：合并后统一变成一个点，其前后的线段需要重新生成
                    # 为了简单起见，我们直接将其变为一个lineTo
                    new_commands.append(('lineTo', center_point))
            # 如果是其他被合并的节点，则直接跳过
            elif is_anchor and anchor_idx_counter in indices_to_merge:
                continue
            # 其他节点照常添加
            else:
                new_commands.append(cmd)

        # 3. 创建并执行命令
        new_path = VectorPath(new_commands)
        new_stroke = canvas._rebuild_stroke_from_path(old_stroke, new_path)

        cmd = ModifyStrokeCommand(self, stroke_index, old_stroke, new_stroke)
        self.undo_stack.push(cmd)

    def on_convert_segment_to_line(self):
        """[槽] 连接到“曲线转直线”按钮。"""
        self.canvas.convert_segment_to_line()

    def on_convert_segment_to_curve(self):
        """[槽] 连接到“直线转曲线”按钮。"""
        self.canvas.convert_segment_to_curve()

    def _update_node_action_buttons(self):
        """[核心][V5.4 最终增强版] 根据画布的详细选择状态，更新所有节点操作按钮的可用性。"""
        # 1. 默认全部禁用
        for btn in self.node_action_buttons.values():
            btn.setEnabled(False)

        canvas = self.canvas
        # 如果没有选中任何笔画，则直接返回
        if not (canvas and canvas.current_char and canvas.selected_stroke_index != -1):
            return

        # 2. 检查笔画索引是否有效
        if not (0 <= canvas.selected_stroke_index < len(canvas.current_char.strokes)):
            return

        # 3. 获取所需的所有上下文信息
        stroke = canvas.current_char.strokes[canvas.selected_stroke_index]
        path = stroke.to_bezier_path()
        selected_indices = canvas.selected_anchor_indices
        num_selected = len(selected_indices)
        
        # 获取路径的总节点数
        num_nodes = 0
        if path.commands:
            num_nodes = sum(1 for cmd in path.commands if cmd[0] in ['moveTo', 'lineTo', 'curveTo'])

        is_closed = path.commands and path.commands[-1][0] == 'closePath'

        # --- 4. 根据上下文启用/禁用按钮 ---

        # a. 总是可用的操作（只要选中了笔画）
        self.node_action_buttons['insert_node'].setEnabled(True)

        # b. 需要至少选中一个节点的操作
        if num_selected > 0:
            self.node_action_buttons['to_corner'].setEnabled(True)
            self.node_action_buttons['to_smooth'].setEnabled(True)
            self.node_action_buttons['to_asymmetric'].setEnabled(True)
            # 只有当节点数大于2时才能删除，防止路径消失
            if num_nodes > 2:
                self.node_action_buttons['delete_node'].setEnabled(True)

        # c. 需要选中多个节点的操作
        if num_selected > 1:
            self.node_action_buttons['merge_nodes'].setEnabled(True)

        # d. 仅当只选择一个节点时，才判断的上下文相关操作
        if num_selected == 1:
            node_idx = next(iter(selected_indices))
            
            # 启用“闭合/断开路径”
            if is_closed:
                # 闭合路径上任意一点都可以作为断开点
                if num_nodes > 2: # 至少需要3个点才能断开
                    self.node_action_buttons['break_path'].setEnabled(True)
            else:
                # 开放路径只有在端点处才能闭合
                if node_idx == 0 or node_idx == num_nodes - 1:
                    self.node_action_buttons['close_path'].setEnabled(True)

            # 智能启用“曲线/直线转换”
            # 找到选中节点 *之后* 的那条线段对应的命令
            cmd_idx, _, _ = canvas._find_command_for_anchor(path.commands, node_idx)
            segment_cmd_index = cmd_idx + 1

            if segment_cmd_index < len(path.commands):
                segment_cmd = path.commands[segment_cmd_index]
                if segment_cmd[0] == 'curveTo':
                    self.node_action_buttons['curve_to_line'].setEnabled(True)
                elif segment_cmd[0] == 'lineTo':
                    self.node_action_buttons['line_to_curve'].setEnabled(True)
    def _on_grid_options_changed(self):
        """
        [槽函数][最终增强版] 当任何显示选项复选框的状态改变时调用。
        
        此方法作为一个统一的处理器，负责从UI界面收集所有与网格显示
        相关的配置（网格、米字格、西文度量线、专业中文辅助线），
        然后调用 DrawingCanvas 的公共接口来更新其内部的显示状态，并触发重绘。
        这是实现UI控件与画布视图解耦的关键一环。
        """
        # --- 1. 安全性检查 ---
        # 确保 self.canvas 对象已创建且仍然存在，防止在程序启动或关闭
        # 过程中的早期/晚期调用引发错误。
        if not hasattr(self, 'canvas') or not self.canvas:
            return

        # --- 2. 从所有相关的UI控件中读取当前状态 ---
        # 通过调用 isChecked() 方法，获取每个复选框的布尔值状态。
        show_grid = self.show_grid_checkbox.isChecked()
        show_guides = self.show_guides_checkbox.isChecked()
        show_metrics = self.show_metrics_checkbox.isChecked()
        # [核心修正] 读取新增的“专业中文辅助线”复选框的状态
        show_pro_guides = self.show_pro_guides_checkbox.isChecked()
        
        # --- 3. 调用 DrawingCanvas 的公共接口来更新视图 ---
        # 将收集到的所有状态作为参数，一次性调用 canvas 的 set_grid_options 方法。
        # DrawingCanvas 内部会处理具体的重绘逻辑。
        self.canvas.set_grid_options(
            show_grid=show_grid, 
            show_guides=show_guides, 
            show_metrics=show_metrics,
            show_pro_guides=show_pro_guides # [核心修正] 传递新的状态参数
        )
    def _on_ai_style_changed(self, value: int):
        """[槽] 当 AI 风格强度滑块变化时调用。"""
        style_strength = value / 100.0
        self.ai_style_label.setText(f"{style_strength:.1f}")

    def on_ai_generate(self):
        """[槽] 启动 AI 生成过程。"""
        prompt = self.ai_prompt_text.toPlainText().strip()
        if not self.current_char_obj:
            QMessageBox.warning(self, "AI 助手", "请先选择一个要进行 AI 设计的字符。")
            return
        if not prompt:
            QMessageBox.warning(self, "AI 助手", "请输入您的创意描述 (Prompt)。")
            return
            
        print(f"启动AI生成: 字符='{self.current_char_obj.char}', Prompt='{prompt}', 强度={self.ai_style_slider.value() / 100.0}")
        # TODO: 在此启动后台线程执行 AI 生成任务
        self.ai_generate_button.setEnabled(False)
        self.ai_status_label.setText("AI 正在创作中...")

    def on_component_selected(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """[槽] 当部件列表的选择变化时，更新预览和按钮状态。"""
        is_selected = current_item is not None
        self.insert_component_btn.setEnabled(is_selected)
        self.delete_component_btn.setEnabled(is_selected)

        if is_selected:
            component_name = current_item.text()
            component = self.components.get(component_name)
            if component:
                pixmap = component.get_preview_image(size=self.component_preview.width())
                self.component_preview.setPixmap(pixmap)
        else:
            self.component_preview.setText("在列表中选择一个部件以预览")

    def on_save_as_component(self):
        """[槽] 将当前字符的设计保存为一个新部件。"""
        print("保存为部件...") # TODO: 实现逻辑

    def on_insert_component(self):
        """[槽] 将选中的部件插入到当前字符。"""
        print("插入部件...") # TODO: 实现逻辑

    def on_delete_component(self):
        """[槽] 删除选中的部件。"""
        print("删除部件...") # TODO: 实现逻辑
    
    
    def _on_pressure_changed(self, state: int):
        """[槽] 当“启用压感”复选框状态改变时调用。"""
        # is_enabled = state == Qt.Checked
        # TODO: 将这个状态传递给绘图逻辑以模拟压感
        print(f"压感效果已 {'启用' if state == Qt.Checked else '禁用'}")

    def _on_antialias_changed(self, state: int):
        """[槽] 当“抗锯齿”复选框状态改变时调用。"""
        # is_enabled = state == Qt.Checked
        # TODO: 更新 DrawingCanvas 的渲染提示
        # self.canvas.set_antialiasing(is_enabled)
        # 在当前实现中，抗锯齿是默认开启的
        print(f"抗锯齿(预览)已 {'启用' if state == Qt.Checked else '禁用'}")

    def on_move_layer_up(self):
        """[槽] 将选中的图层上移一层。"""
        if not self.current_char_obj or not self.layer_list_widget.currentItem():
            return
        
        current_row = self.layer_list_widget.currentRow()
        if current_row > 0: # 确保不是最顶层
            # TODO: 实现 QUndoCommand
            # 列表是倒序的，上移等于在模型中下移
            model_index_from = len(self.current_char_obj.strokes) - 1 - current_row
            model_index_to = model_index_from + 1
            
            stroke = self.current_char_obj._strokes.pop(model_index_from)
            self.current_char_obj._strokes.insert(model_index_to, stroke)
            
            self.is_project_dirty = True
            self._update_all_views()
            # 保持选中
            self.layer_list_widget.setCurrentRow(current_row - 1)
    def on_move_layer_down(self):
        """[槽] 将选中的图层下移一层。"""
        if not self.current_char_obj or not self.layer_list_widget.currentItem():
            return
            
        current_row = self.layer_list_widget.currentRow()
        if current_row < self.layer_list_widget.count() - 1: # 确保不是最底层
            # TODO: 实现 QUndoCommand
            # 列表是倒序的，下移等于在模型中上移
            model_index_from = len(self.current_char_obj.strokes) - 1 - current_row
            model_index_to = model_index_from - 1
            
            stroke = self.current_char_obj._strokes.pop(model_index_from)
            self.current_char_obj._strokes.insert(model_index_to, stroke)

            self.is_project_dirty = True
            self._update_all_views()
            # 保持选中
            self.layer_list_widget.setCurrentRow(current_row + 1)
    def _update_layer_buttons_state(self):
        """[槽][已升级] 当图层列表的选择变化时，更新图层操作按钮和透明度滑块的状态。"""
        selected_items = self.layer_list_widget.selectedItems()
        has_selection = bool(selected_items)
        is_single_selection = len(selected_items) == 1
        
        self.del_layer_btn.setEnabled(has_selection)
        self.up_layer_btn.setEnabled(is_single_selection)
        self.down_layer_btn.setEnabled(is_single_selection)
        
        # [核心新增] 更新透明度滑块状态
        self.layer_opacity_slider.setEnabled(is_single_selection)
        if is_single_selection:
            stroke = self._get_selected_stroke()
            if stroke:
                # 更新滑块时暂时断开信号，防止触发 valueChanged
                self.layer_opacity_slider.blockSignals(True)
                self.layer_opacity_slider.setValue(int(stroke.opacity * 100))
                self.layer_opacity_label.setText(f"{int(stroke.opacity * 100)}%")
                self.layer_opacity_slider.blockSignals(False)
        else:
            self.layer_opacity_label.setText("--")
    def _on_tool_selected(self):
        """
        [槽][已修正] 当一个工具按钮被点击时调用。
        现在 QButtonGroup 会自动处理UI高亮，此函数只负责更新逻辑状态。
        """
        # 1. 获取当前被选中的按钮
        # QButtonGroup 会自动管理哪个按钮是 checked 状态。
        checked_button = self.tool_button_group.checkedButton()
        if not checked_button:
            return

        # 2. [核心修正] 状态管理：自动退出“插入节点”模式
        #    如果当前处于“插入节点”模式 (按钮被按下)，但用户又点击了
        #    一个非“节点编辑”的工具按钮，那么就应该自动取消“插入节点”模式。
        if self.insert_node_btn.isChecked() and checked_button is not self.tool_buttons['node_editor']:
            # blockSignals(True) 暂时阻止信号发射，防止 on_toggle_insert_node_mode 被重复调用
            self.insert_node_btn.blockSignals(True)
            self.insert_node_btn.setChecked(False) # 取消按钮的选中状态
            self.insert_node_btn.blockSignals(False)
            
            # 手动更新画布状态，因为信号被阻塞了
            if self.canvas:
                self.canvas.is_insert_mode = False

        # 3. 更新当前激活的工具ID
        tool_id = checked_button.property("tool_id")
        self.current_tool = tool_id
        
        # 4. 根据新工具更新画布的光标样式
        if self.canvas:
            if tool_id == 'node_editor':
                self.canvas.setCursor(Qt.ArrowCursor) # 节点编辑默认用箭头光标
            elif tool_id == 'eraser':
                self.canvas.setCursor(Qt.PointingHandCursor) # 橡皮擦用手型光标
            else:
                self.canvas.setCursor(Qt.CrossCursor) # 其他绘图工具用十字光标
            
            # 5. [重要] 刷新画布以显示/隐藏节点
            #    如果切换到/切换出节点编辑模式，需要立即重绘画布
            #    来正确地显示或隐藏路径上的节点和控制柄。
            self.canvas.update()
            
        # 6. 更新状态栏的文本提示
        self.status_label.setText(f"当前工具: {checked_button.text()}")
    
    def _on_stroke_width_changed(self, value: int):
        """[槽] 当笔画宽度滑块变化时调用。"""
        self.stroke_width = value
        self.stroke_width_label.setText(str(value))

    def _on_smoothing_changed(self, value: int):
        """[槽] 当平滑度滑块变化时调用。"""
        self.stroke_smoothing = value / 100.0
        self.smoothing_label.setText(f"{self.stroke_smoothing:.1f}")
        
    def _on_color_btn_clicked(self):
        """[槽] 当颜色选择按钮被点击时调用。"""
        new_color = QColorDialog.getColor(self.current_color, self, "选择笔画颜色")
        if new_color.isValid():
            self.current_color = new_color
            self._update_color_btn_style()

    def on_clear_char(self):
        """[槽] 清空当前字符的所有笔画。"""
        if not self.current_char_obj: return
        reply = QMessageBox.question(self, "确认清空", 
            f"您确定要清空字符 '{self.current_char_obj.char}' 的所有设计吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # TODO: 实现 QUndoCommand for clearing all strokes
            self.current_char_obj.clear_all_layers()
            self.is_project_dirty = True
            self._update_all_views()

    def _update_color_btn_style(self):
        """[私有辅助] 更新颜色按钮的背景色。"""
        self.color_btn.setStyleSheet(f"background-color: {self.current_color.name()}; border: 1px solid #ccc;")

    def _init_tool_state(self):
        """[私有辅助] 在启动或新建项目后初始化工具面板的状态。"""
        self.tool_buttons[self.current_tool].setChecked(True)
        self.stroke_width_slider.setValue(self.stroke_width)
        self.smoothing_slider.setValue(int(self.stroke_smoothing * 100))
        self._update_color_btn_style()
    def on_load_database(self):
        if not self._confirm_dirty_project("加载新字库"): return
        filename, _ = QFileDialog.getOpenFileName(self, "选择基础字符数据库", "", "SQLite 数据库 (*.db)")
        if filename:
            try:
                self.load_project_data(filename, is_db=True)
                self.current_project_path = None
                self.is_project_dirty = False
                self.undo_stack.clear()
                self.setWindowTitle("未命名项目 - MCDCNFD")
                self.statusBar().showMessage(f"已从 '{os.path.basename(filename)}' 加载新字库。", 5000)
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载数据库时发生错误:\n{e}")

    def on_new_project(self):
        if self._confirm_dirty_project("新建项目"):
            self.load_project_data("font_char_data.db", is_db=True)
            self.current_project_path = None; self.is_project_dirty = False
            self.undo_stack.clear(); self.setWindowTitle("中文字体设计 - MCDCNFD- 作者：跳舞的火公子")

    def on_open_project(self):
        if not self._confirm_dirty_project("打开项目"): return
        filename, _ = QFileDialog.getOpenFileName(self, "打开字体项目", "", "字体项目文件 (*.mcdcnfd)")
        if filename: self.load_project_data(filename)
    
    def on_save_project(self):
        if not self.current_project_path: self.on_save_project_as()
        else: self._save_to_file(self.current_project_path)

    def on_save_project_as(self):
        filename, _ = QFileDialog.getSaveFileName(self, "项目另存为", "", "字体项目文件 (*.mcdcnfd)")
        if filename: self._save_to_file(filename)

    def _save_to_file(self, filename):
        """
        [已增强] 将当前项目的所有数据（字符、部件、设置和字偶距）保存到指定的 .mcdcnfd 文件中。
        """
        self.statusBar().showMessage(f"正在保存项目到 {os.path.basename(filename)}...")
        QApplication.processEvents() # 确保状态栏消息立即显示

        try:
            # 1. 序列化所有已设计的字符
            font_chars_data = {
                char_str: char_obj.to_dict()
                for char_str, char_obj in self.font_chars.items()
                if char_obj.is_designed
            }

            # 2. 序列化所有部件
            components_data = {comp.name: comp.to_dict() for comp in self.components.values()}
            
            # 3. 构建顶层的项目数据字典
            project_data = {
                'file_format_version': '1.0-qt',
                'saved_at': datetime.now().isoformat(),
                
                # 保存字体全局设置
                'font_settings': self.font_settings,
                
                # [核心新增] 保存字偶距数据字典
                'kerning_pairs': self.kerning_pairs,
                
                'components': components_data,
                'font_chars': font_chars_data,
            }

            # 4. 将字典以JSON格式写入文件
            with open(filename, 'w', encoding='utf-8') as f:
                # indent=2 使保存的JSON文件具有缩进，易于人类阅读和调试
                json.dump(project_data, f, ensure_ascii=False, indent=2)

            # 5. 更新程序状态
            self.current_project_path = filename
            self.is_project_dirty = False # 保存后，项目不再是“脏”的
            self.setWindowTitle(f"{os.path.basename(filename)} - MCDCNFD")
            self.statusBar().showMessage("项目已成功保存！", 5000)
            
        except Exception as e:
            # 如果发生任何错误，则显示一个严重错误对话框
            QMessageBox.critical(self, "保存失败", f"保存项目时发生错误: {e}")
            self.statusBar().showMessage(f"保存失败: {e}", 5000)

    def on_export_ttf(self):
        if not FONTTOOLS_AVAILABLE:
            QMessageBox.warning(self, "依赖缺失", "需要安装`fonttools`库才能导出TTF字体。\n\n请运行: `pip install fonttools`")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出TTF字体", "", "TrueType 字体 (*.ttf)")
        if not filename: return

        font_metadata = {
            'units_per_em': 1024,
            'name_table': {'familyName': 'My Handwriting Font', 'styleName': 'Regular'},
            'os2_table': {'sTypoAscender': 824, 'sTypoDescender': -200, 'usWinAscent': 1000, 'usWinDescent': 200},
            'hhea_table': {'ascent': 824, 'descent': -200, 'lineGap': 0}
        }
        export_options = {'subsetting': True}
        
        designed_chars_count = sum(1 for c in self.font_chars.values() if c.is_designed)
        self.progress_dialog = QProgressDialog("正在导出字体...", "取消", 0, designed_chars_count, self)
        self.progress_dialog.setWindowTitle("导出进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        
        self.export_worker = ExportWorker(filename, list(self.font_chars.values()), font_metadata, {}, export_options)
        self.export_worker.signals.progress.connect(self.on_export_progress)
        self.export_worker.signals.finished.connect(self.on_export_finished)
        self.export_worker.signals.error.connect(self.on_export_error)
        self.progress_dialog.canceled.connect(self.export_worker.cancel)
        
        QThreadPool.globalInstance().start(self.export_worker)
        self.progress_dialog.show()

    def on_export_progress(self, current, total, char_name):
        if self.progress_dialog:
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(f"正在处理 '{char_name}' ({current + 1}/{total})")

    def on_export_finished(self, filename):
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.information(self, "导出成功", f"字体已成功导出到:\n{filename}")
        self.statusBar().showMessage("字体导出成功！", 5000)

    def on_export_error(self, error_message):
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "导出失败", error_message)
        self.statusBar().showMessage("字体导出失败！", 5000)

    def _perform_search(self, *args): # 添加 *args 以接收任何信号的参数
        """
        根据所有过滤器更新字符列表。
        [已修正] 从 QCheckBox 获取设计状态。
        """
        query = self.search_box.text().strip().lower()
        category = self.category_combo.currentText()
        
        # *** 关键修正 ***
        # 从复选框获取状态
        show_designed = self.designed_checkbox.isChecked()
        show_undesigned = self.undesigned_checkbox.isChecked()
        
        filtered_chars = []
        for char_obj in self.font_chars.values():
            # 分类过滤
            if category != "全部" and char_obj.category != category:
                continue
            
            # 搜索过滤
            if query and not (query in char_obj.char or query in char_obj.pinyin.lower()):
                continue
            
            # 设计状态过滤
            is_char_designed = char_obj.is_designed
            if not show_designed and is_char_designed:
                continue
            if not show_undesigned and not is_char_designed:
                continue
            
            filtered_chars.append(char_obj)
            
        self.char_list_model.set_characters(filtered_chars)
        
    def on_char_selected(self, index: QModelIndex):
        """[已修复] 当用户在列表中选择一个字符时的处理函数。"""
        char_obj: FontChar = self.char_list_model.get_char_obj_by_index(index)
        if char_obj and self.current_char_obj != char_obj:
            self.current_char_obj = char_obj
            self.undo_stack.clear()

            # [核心修复] 切换字符时，必须重置所有与子选择相关的状态。
            if self.canvas:
                self.canvas.selected_stroke_index = -1
                self.canvas.selected_anchor_indices.clear()
                self.canvas.active_anchor_index = -1
                self.canvas.selected_segment_index = None
                self.canvas.dragged_node_info = None
                self.canvas.marquee_start_pos = None
                self.canvas.marquee_rect = None

            self._update_all_views()
            
            if self.char_as_ref_checkbox.isChecked():
                self.on_toggle_reference_image()
    def on_toggle_reference_image(self):
        """[槽][已修正] 当“参考底模”复选框状态改变时调用。"""
        is_checked = self.char_as_ref_checkbox.isChecked()
        
        if is_checked and self.current_char_obj:
            self.statusBar().showMessage(f"正在为 '{self.current_char_obj.char}' 生成参考底模...")
            
            canvas_pixel_size = self.canvas._pixel_size * self.canvas.grid_size
            if canvas_pixel_size < 1: canvas_pixel_size = 512 # Fallback
            
            # [核心修正] 从 canvas 获取中宫比例，并传递给 Worker
            zhonggong_scale = self.canvas.zhonggong_scale
            worker = ReferenceImageWorker(
                self.current_char_obj.char, 
                int(canvas_pixel_size),
                zhonggong_scale=zhonggong_scale 
            )

            worker.signals.finished.connect(self.on_reference_image_ready)
            QThreadPool.globalInstance().start(worker)
        else:
            self.canvas.set_reference_image(None)

    def on_reference_image_ready(self, pixmap: QPixmap):
        """[槽] 当后台工作器生成好参考图像后调用。"""
        self.canvas.set_reference_image(pixmap)
        self.statusBar().showMessage("参考底模已加载。", 3000)
    def on_stroke_finished(self, new_stroke: HandwritingStroke):
        if self.current_char_obj:
            command = AddStrokeCommand(self.current_char_obj, new_stroke, self)
            self.undo_stack.push(command)
    def on_stroke_modified(self, stroke_index: int, old_stroke: HandwritingStroke, new_stroke: HandwritingStroke):
        """
        [槽函数] 当 DrawingCanvas 完成一次节点编辑后调用。
        
        此函数负责创建 ModifyStrokeCommand 并将其推入撤销栈。
        """
        if self.current_char_obj:
            command = ModifyStrokeCommand(
                self,           # 传递 MainWindow 的引用
                stroke_index,
                old_stroke,
                new_stroke
            )
            self.undo_stack.push(command)
            # push 命令会自动调用 redo()，而 redo() 内部会调用 _update_all_views()
            # 所以这里不需要再手动刷新视图。

    def on_stroke_finished(self, new_stroke: HandwritingStroke):
        if self.current_char_obj:
            # 注意：这里需要修改 AddStrokeCommand 的构造函数调用
            command = AddStrokeCommand(self.current_char_obj, new_stroke, self)
            self.undo_stack.push(command)
    def on_add_layer(self):
        if self.current_char_obj:
            stroke = HandwritingStroke()
            command = AddStrokeCommand(self.current_char_obj, stroke, self)
            self.undo_stack.push(command)

    def on_remove_layer(self):
        if self.current_char_obj and self.layer_list_widget.currentItem():
            index = self.layer_list_widget.currentRow()
            # 注意图层列表是倒序的
            actual_index = len(self.current_char_obj.strokes) - 1 - index
            command = RemoveStrokeCommand(self.current_char_obj, actual_index, self)
            self.undo_stack.push(command)

    def _update_all_views(self):
        """
        [控制器核心][V5.5 属性面板增强版] 在模型数据发生变化后，统一刷新所有相关的视图组件。

        此版本新增了对“属性”面板的更新调用，确保在每次选择变化或数据
        修改后，属性面板也能正确地反映当前上下文。
        """
        # --- 1. 更新宏观状态：窗口标题和状态栏 ---
        char_name = self.current_char_obj.char if self.current_char_obj else "无"
        dirty_indicator = "*" if self.is_project_dirty else ""
        project_name = os.path.basename(self.current_project_path) if self.current_project_path else "未命名项目"
        
        self.setWindowTitle(f"{dirty_indicator}{project_name} - MCDCNFD")
        
        stats = self.get_stats()
        self.status_label.setText(
            f"当前字符: {char_name} | 已设计: {stats['designed']}/{stats['total']} ({stats['progress']:.1f}%) | {dirty_indicator}{project_name}"
        )
        if hasattr(self, 'navbar_stats_label'):
             self.navbar_stats_label.setText(f"总字符: {stats['total']} | 已设计: {stats['designed']} | 进度: {stats['progress']:.1f}%")

        # --- 2. 更新中央面板的标题区 ---
        if self.current_char_obj:
            char_obj = self.current_char_obj
            self.header_char_display.setText(char_obj.char if char_obj.char != ' ' else 'SP')
            self.header_char_label.setText(f"正在编辑: {char_obj.char}")
            info_text = (f"Unicode: U+{char_obj.unicode_val:04X} | "
                         f"笔画数: {len(char_obj.strokes)} (绘) / {char_obj.stroke_count_db or 'N/A'} (预估) | "
                         f"拼音: {char_obj.pinyin or 'N/A'}")
            self.header_char_info_label.setText(info_text)
        else:
            self.header_char_display.setText('')
            self.header_char_label.setText('请从左侧选择一个字符开始编辑')
            self.header_char_info_label.setText('Unicode: | 笔画数: | 拼音:')
            
        # --- 3. 更新所有核心视图和面板 ---
        
        # 3.1 更新中央画布
        self.canvas.set_char(self.current_char_obj)
        
        # 3.2 更新图层列表
        self.update_layer_list()
        
        # [核心新增] 3.3 更新属性面板
        # 这个调用会根据画布的当前选择状态，动态显示或隐藏相关控件
        self._update_properties_panel()
        
        # 3.4 更新所有预览面板
        self.update_previews()
        
        # 3.5 更新分析面板
        self.update_stats_display()
        
        # --- 4. 更新工具栏按钮的状态 ---
        
        # 4.1 更新节点编辑相关的按钮状态
        self.update_node_tool_buttons()
        
        # 4.2 更新“字形矢量化”按钮的状态
        self.vectorize_glyph_btn.setEnabled(self.current_char_obj is not None)
        
        # --- 5. 刷新字符浏览器 ---
        # 强制字符列表视图重绘，以正确反映“已设计”状态和当前选中项
        self.char_list_view.update()
        
        # --- 6. 同步参考底图状态 ---
        if self.char_as_ref_checkbox.isChecked():
             self.on_toggle_reference_image()
        else:
             self.canvas.set_reference_image(None)
    def on_segment_tension_changed(self):
        """[槽][新增] 当张力滑块释放时调用，固化线段张力修改。"""
        canvas = self.canvas
        if not canvas or canvas.selected_segment_index is None or not canvas.pre_modification_stroke:
            return

        # 获取最终的张力值
        new_tension_ratio = self.segment_tension_slider.value() / 100.0
        
        # 使用拖动前保存的笔画状态作为修改基础
        old_stroke = canvas.pre_modification_stroke
        path = old_stroke.to_bezier_path()
        commands = list(path.commands)
        seg_idx = canvas.selected_segment_index

        if 0 < seg_idx < len(commands) and commands[seg_idx][0] == 'curveTo':
            cmd = commands[seg_idx]
            p0 = np.array(canvas._get_anchor_point_before(commands, seg_idx))
            p1 = np.array(cmd[3])
            c1_orig = np.array(cmd[1])
            c2_orig = np.array(cmd[2])

            # 计算方向向量
            v1 = c1_orig - p0
            v2 = c2_orig - p1
            
            # 计算新长度
            base_length = np.linalg.norm(p1 - p0)
            new_len1 = base_length * new_tension_ratio
            new_len2 = np.linalg.norm(v2) # 保持第二个控制柄的绝对长度不变，或也可以按比例缩放

            # 计算新控制点
            norm_v1 = v1 / (np.linalg.norm(v1) + 1e-9)
            norm_v2 = v2 / (np.linalg.norm(v2) + 1e-9)
            new_c1 = tuple(p0 + norm_v1 * new_len1)
            new_c2 = tuple(p1 + norm_v2 * new_len2)
            
            commands[seg_idx] = ('curveTo', new_c1, new_c2, tuple(p1))

            new_path = VectorPath(commands)
            rebuilt_stroke = canvas._rebuild_stroke_from_path(old_stroke, new_path)
            
            # 创建并提交命令
            command = ModifyStrokeCommand(self, canvas.selected_stroke_index, old_stroke, rebuilt_stroke)
            self.undo_stack.push(command)

    def update_layer_list(self):
        """[已升级] 使用自定义 Widget 填充图层列表，并连接其内部信号。"""
        self.layer_list_widget.clear()
        if not self.current_char_obj:
            return

        for i in range(len(self.current_char_obj.strokes) - 1, -1, -1):
            stroke = self.current_char_obj.strokes[i]
            
            item = QListWidgetItem(self.layer_list_widget)
            item_widget = LayerItemWidget(stroke)
            
            # 存储笔画在模型中的真实索引
            item.setData(Qt.UserRole, i)
            
            # [核心] 连接自定义 Widget 的信号到 MainWindow 的槽函数
            # 使用 lambda 传递索引
            item_widget.visibility_changed.connect(lambda checked, index=i: self.on_layer_visibility_changed(index, checked))
            item_widget.lock_changed.connect(lambda checked, index=i: self.on_layer_lock_changed(index, checked))

            item.setSizeHint(item_widget.sizeHint())
            self.layer_list_widget.addItem(item)
            self.layer_list_widget.setItemWidget(item, item_widget)
            
        self._update_layer_buttons_state()   
    def update_previews(self):
        """[已完善] 统一更新所有预览视图（单字预览和文本行预览）。"""
        # --- 1. 更新单字预览 ---
        if self.current_char_obj:
            # 调用字符对象的 get_preview_image 方法，获取一个256x256的预览图
            pixmap = self.current_char_obj.get_preview_image(all_components=self.components, size=256)
            self.preview_label.setPixmap(pixmap)
        else:
            # 如果没有选中字符，则清空预览区并显示提示文字
            self.preview_label.clear()
            self.preview_label.setText("选择字符以预览")
            
        # --- 2. 更新文本行预览 ---
        # 确保 text_preview_widget 已经被创建
        if hasattr(self, 'text_preview_widget'):
            # 获取输入框中的文本
            text = self.preview_text_edit.text()
            # 调用自定义控件的 set_text 方法，这会触发其内部的 paintEvent
            self.text_preview_widget.set_text(text)
            
    def _confirm_dirty_project(self, title: str) -> bool:
        """如果项目有未保存更改，则询问用户。返回True表示可以继续。"""
        if not self.is_project_dirty: return True
        reply = QMessageBox.question(self, title, '有未保存的更改，是否要保存？',
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if reply == QMessageBox.Save: self.on_save_project()
        return reply != QMessageBox.Cancel

    def closeEvent(self, event: "QCloseEvent"):
        if self._confirm_dirty_project("退出"):
            event.accept()
        else:
            event.ignore()

    def load_project_data(self, path, is_db=False):
        """
        [已增强] 加载项目数据（从.db或.mcdcnfd文件）并填充UI的通用方法。
        
        此版本新增了从项目文件中加载字偶距数据，并更新“字跨”面板UI的功能。
        """
        self.statusBar().showMessage(f"正在加载 {os.path.basename(path)}...")
        QApplication.processEvents() # 确保状态栏消息立即显示
        
        # --- 步骤 1: 清理当前项目状态 ---
        self.font_chars.clear()
        self.current_char_obj = None

        # --- 步骤 2: 加载基础字符数据库 ---
        base_db_path = "font_char_data.db"
        try:
            if is_db:
                self.data_manager.load_database_from_file(path)
            else:
                self.data_manager.load_database_from_file(base_db_path)
        except (FileNotFoundError, RuntimeError) as e:
            QMessageBox.critical(self, "数据库加载失败", f"无法加载基础字符数据库 '{base_db_path}':\n{e}\n应用程序无法继续。")
            self.close()
            return
            
        # --- 步骤 3: 根据数据库，初始化所有 FontChar 对象 ---
        all_chars_data = self.data_manager.get_all_chars_data()
        for char_data in all_chars_data:
            char = char_data.get('char')
            if char:
                self.font_chars[char] = FontChar(char, char_data, self.grid_size, self.data_manager)

        # --- 步骤 4: 如果是加载项目文件，则覆盖设计数据和设置 ---
        if not is_db:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)

                # a. 加载字体设置
                self.font_settings.update(project_data.get('font_settings', {}))
                
                # b. [核心新增] 加载字偶距数据
                self.kerning_pairs = project_data.get('kerning_pairs', {})
                
                # c. 加载已设计的字符数据
                for char_str, char_dict in project_data.get('font_chars', {}).items():
                    if char_str in self.font_chars:
                        self.font_chars[char_str] = FontChar.from_dict(char_dict, self.data_manager, self.grid_size)
                
                self.current_project_path = path
            except Exception as e:
                QMessageBox.critical(self, "项目文件加载失败", f"无法解析项目文件 '{path}':\n{e}")
                return

        # --- 步骤 5: 更新UI ---
        # 更新字符浏览器的分类下拉框
        categories = ["全部"] + sorted(list(set(c.category for c in self.font_chars.values() if c.category)))
        self.category_combo.clear()
        self.category_combo.addItems(categories)
        
        # 更新“设置”面板的UI
        self._update_settings_ui_from_data()
        
        # [核心新增] 加载数据后，刷新字偶距列表
        self._update_kerning_list()
        
        # 刷新字符列表
        self._perform_search()
        
        # 刷新所有其他视图（画布、预览、分析等）
        self._update_all_views()
        
        self.statusBar().showMessage("加载完成。", 5000)

    def _update_settings_ui_from_data(self):
        """[新增辅助] 从 self.font_settings 字典加载数据来更新设置面板的UI控件。"""
        # 暂时断开信号连接，避免在程序设置UI时触发 on-change 逻辑
        for widget in self.findChildren(QWidget):
            if "setting_" in widget.objectName():
                widget.blockSignals(True)
        
        # 更新UI控件的值
        self.setting_family_name.setText(self.font_settings['familyName'])
        self.setting_style_name.setText(self.font_settings['styleName'])
        self.setting_version.setText(self.font_settings['version'])
        self.setting_copyright.setText(self.font_settings['copyright'])
        self.setting_upm_combo.setCurrentText(str(self.font_settings['unitsPerEm']))
        self.setting_ascender_slider.setValue(self.font_settings['ascender'])
        self.setting_descender_slider.setValue(self.font_settings['descender'])
        self.setting_xheight_slider.setValue(self.font_settings['xHeight'])
        self.setting_capheight_slider.setValue(self.font_settings['capHeight'])

        # 重新连接信号
        for widget in self.findChildren(QWidget):
            if "setting_" in widget.objectName():
                widget.blockSignals(False)

    def get_stats(self) -> Dict[str, Any]:
        """[控制器辅助] 计算当前项目的统计信息。"""
        total = len(self.font_chars)
        if total == 0:
            return {'total': 0, 'designed': 0, 'progress': 0.0}
        
        designed = sum(1 for char_obj in self.font_chars.values() if char_obj.is_designed)
        progress = (designed / total * 100) if total > 0 else 0.0
        return {'total': total, 'designed': designed, 'progress': progress}
    def update_stats_display(self):
        """
        [已完善] 更新“分析”标签页中的信息统计文本区域。
        
        此方法会汇总项目级的宏观统计数据和当前字符的微观信息，
        并使用 HTML 格式化后显示在 QTextEdit 中，以实现富文本效果。
        """
        # --- 安全检查 ---
        # 确保 UI 组件已创建且仍然存在
        if not hasattr(self, 'stats_text') or not self.stats_text.isVisible():
            return

        # --- 1. 获取项目宏观统计数据 ---
        stats = self.get_stats()
        
        # --- 2. 使用 HTML 字符串构建显示内容 ---
        # 使用 <style> 标签来定义通用样式，使HTML更简洁
        # 使用 <b> 标签加粗标题，<hr> 添加分割线
        content = f"""
        <style>
            p {{ font-size: 11pt; color: #334155; }}
            b {{ color: #0f172a; }}
            hr {{ border: 1px solid #e2e8f0; }}
        </style>
        <p>
            <b>项目进度: {stats['progress']:.1f}%</b><br>
            - 已设计: {stats['designed']}<br>
            - 总字符: {stats['total']}
        </p>
        <hr>
        """
        
        # --- 3. 如果有选中的字符，则添加其详细信息 ---
        if self.current_char_obj:
            char_obj = self.current_char_obj
            # 计算边界框信息
            bounds = char_obj.get_bounds(self.components)
            if bounds:
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                bounds_text = f"{width:.1f} x {height:.1f}"
            else:
                bounds_text = "N/A"

            content += f"""
            <p>
                <b>当前字符: '{char_obj.char}'</b><br>
                - Unicode: U+{char_obj.unicode_val:04X}<br>
                - 笔画数 (绘制): {len(char_obj.strokes)}<br>
                - 笔画数 (预估): {char_obj.stroke_count_db or 'N/A'}<br>
                - 分类: {char_obj.category or 'N/A'}<br>
                - 拼音: {char_obj.pinyin or 'N/A'}<br>
                - 边界框 (W x H): {bounds_text}
            </p>
            """
        else:
            content += """
            <p style="color: #9ca3af;">没有选中任何字符。</p>
            """
        
        # --- 4. 将格式化后的 HTML 内容设置到 QTextEdit ---
        self.stats_text.setHtml(content)





    def on_vectorize_glyph(self):
        """
        [槽函数][最终增强版] 当“字形矢量化”按钮被点击时调用。
        此版本能够正确处理多轮廓字形，将其分解为多个独立的笔画图层。
        """
        if not self.current_char_obj:
            return
        if not FONTTOOLS_AVAILABLE:
            QMessageBox.warning(self, "依赖缺失", "此功能需要 `fontTools` 库。\n请运行: pip install fonttools")
            return

        reply = QMessageBox.question(self, "确认操作", 
            f"您要将字符 '{self.current_char_obj.char}' 的标准字形转换为矢量笔画吗？\n"
            f"字形的每个独立部分都将作为一个新图层添加到当前设计中。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.No:
            return

        self.statusBar().showMessage("正在矢量化字形...")
        QApplication.processEvents()

        try:
            # 1. 从字体文件中提取包含所有轮廓的完整矢量路径
            vector_path = self._get_vector_path_from_font(self.current_char_obj.char)
            
            if vector_path and not vector_path.is_empty():
                # 2. [核心修正] 将完整路径分解为独立的子路径（每个子路径对应一个轮廓）
                sub_paths = vector_path.get_subpaths()
                
                new_strokes = []
                # 3. 遍历每个子路径，并为每一个都创建一个 HandwritingStroke 对象
                for sub_path in sub_paths:
                    if not sub_path.is_empty():
                        stroke = HandwritingStroke.from_vector_path(sub_path, self.current_color.name())
                        new_strokes.append(stroke)
                
                # 4. 如果成功生成了任何笔画，则使用新的命令将其添加到撤销栈
                if new_strokes:
                    # 使用新的 AddMultipleStrokesCommand 来确保整个操作是可撤销的
                    command = AddMultipleStrokesCommand(self.current_char_obj, new_strokes, self)
                    self.undo_stack.push(command)
                    self.statusBar().showMessage("字形矢量化成功！", 3000)
                else:
                    QMessageBox.warning(self, "矢量化失败", f"无法从 '{self.current_char_obj.char}' 的字形中提取有效的轮廓。")
                    self.statusBar().showMessage("矢量化失败，未找到有效轮廓。", 3000)
            else:
                QMessageBox.warning(self, "矢量化失败", f"无法找到或解析字符 '{self.current_char_obj.char}' 的字形。")
                self.statusBar().showMessage("矢量化失败。", 3000)

        except Exception as e:
            QMessageBox.critical(self, "矢量化出错", f"处理字体文件时发生错误: {e}")
            self.statusBar().showMessage(f"矢量化出错: {e}", 5000)
            import traceback
            traceback.print_exc()


    def _find_system_font(self, char_to_check: str) -> Optional[str]:
        """
        [私有辅助][新增] 尝试在系统中查找一个包含指定字符的可用中文字体文件。
        这是一个简化的实现，按预定义的顺序查找常见字体。
        """
        system = platform.system()
        font_paths_to_try = []

        if system == "Windows":
            windir = os.environ.get("WINDIR", "C:\\Windows")
            font_paths_to_try = [
                os.path.join(windir, "Fonts", "msyh.ttc"),  # 微软雅黑 (TTC)
                os.path.join(windir, "Fonts", "msyh.ttf"),  # 微软雅黑 (TTF)
                os.path.join(windir, "Fonts", "simsun.ttc"), # 宋体
            ]
        elif system == "Darwin":  # macOS
            font_paths_to_try = [
                "/System/Library/Fonts/PingFang.ttc",    # 苹方
                "/System/Library/Fonts/STHeiti Medium.ttc", # 华文黑体
                "/System/Library/Fonts/儷黑 Pro.ttf",     # LiHei Pro
            ]
        else:  # Linux
            font_paths_to_try = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
            ]

        for font_path in font_paths_to_try:
            if os.path.exists(font_path):
                # 简单的检查：尝试打开并获取cmap，看是否包含字符
                try:
                    font = TTFont(font_path, fontNumber=0) # 对于TTC，先尝试第一个字体
                    if ord(char_to_check) in font.getBestCmap():
                        return font_path
                    font.close()
                except Exception:
                    continue
        return None


    def _get_vector_path_from_font(self, char: str) -> Optional[VectorPath]:
        """
        [私有辅助][最终修复版 V3 - 彻底绕过兼容性问题] 
        使用 fontTools 从系统字体文件中提取字符的矢量轮廓。
        此版本通过手动计算边界框，完全绕过了在旧版 fontTools 中有问题的 recalcBounds API。
        """
        try:
            font_path = self._find_system_font(char)
            if not font_path:
                raise FileNotFoundError(f"在系统中未找到包含字符 '{char}' 的合适中文字体。")

            font = TTFont(font_path, fontNumber=0)
            glyph_set = font.getGlyphSet()
            
            char_code = ord(char)
            if char_code not in font.getBestCmap():
                font.close()
                return None
            
            glyph_name = font.getBestCmap()[char_code]
            glyph = glyph_set[glyph_name]

            # 1. 使用 RecordingPen 记录字形的绘制命令
            pen = RecordingPen()
            glyph.draw(pen)
            
            path = VectorPath() # 先创建一个空的路径对象

            # 检查是否有绘制内容
            if not pen.value:
                font.close()
                return path # 返回空路径

            # --- [核心修复] 手动从 pen 的记录中计算边界框 ---
            all_points = []
            for _, points in pen.value:
                all_points.extend(points)

            if not all_points:
                font.close()
                return path # 如果有指令但没有点，也返回空路径

            np_points = np.array(all_points)
            xMin, yMin = np_points.min(axis=0)
            xMax, yMax = np_points.max(axis=0)
            bounds = (xMin, yMin, xMax, yMax)
            # --- 边界框计算结束 ---

            upm = font['head'].unitsPerEm
            glyph_width = bounds[2] - bounds[0]
            glyph_height = bounds[3] - bounds[1]
            
            if glyph_width == 0 or glyph_height == 0:
                font.close()
                return path

            # --- 坐标变换 (逻辑保持不变) ---
            target_size = self.grid_size * self.canvas.zhonggong_scale
            scale = target_size / max(glyph_width, glyph_height, 1) # 避免除以零

            final_width = glyph_width * scale
            final_height = glyph_height * scale
            tx = (self.grid_size - final_width) / 2 - (bounds[0] * scale)
            ty = (self.grid_size - final_height) / 2 + (bounds[3] * scale)
            
            transform = Transform(scale, 0, 0, -scale, tx, ty)

            # --- 应用变换并构建 VectorPath (逻辑保持不变) ---
            for command, points in pen.value:
                transformed_points = [transform.transformPoint(p) for p in points]
                
                if command == "moveTo":
                    path.moveTo(*transformed_points[0])
                elif command == "lineTo":
                    path.lineTo(*transformed_points[0])
                elif command == "qCurveTo":
                    path.qCurveTo(transformed_points[0][0], transformed_points[0][1], 
                                  transformed_points[1][0], transformed_points[1][1])
                elif command == "curveTo":
                    path.curveTo(transformed_points[0][0], transformed_points[0][1], 
                                 transformed_points[1][0], transformed_points[1][1], 
                                 transformed_points[2][0], transformed_points[2][1])
                elif command == "closePath":
                    path.closePath()
            
            font.close()
            return path
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def _get_selected_stroke(self) -> Optional['HandwritingStroke']:
        """[私有辅助] 获取当前在图层列表中选中的笔画对象。"""
        if not self.current_char_obj or not self.layer_list_widget.currentItem():
            return None
        item = self.layer_list_widget.currentItem()
        index = item.data(Qt.UserRole)
        if 0 <= index < len(self.current_char_obj.strokes):
            return self.current_char_obj.strokes[index]
        return None

    def on_layer_visibility_changed(self, index: int, is_visible: bool):
        """[槽][已升级Undo] 当用户点击图层可见性图标时，创建命令。"""
        if self.current_char_obj and 0 <= index < len(self.current_char_obj.strokes):
            description = f"隐藏图层 '笔画 {index + 1}'" if not is_visible else f"显示图层 '笔画 {index + 1}'"
            command = ModifyStrokePropertyCommand(self, index, 'is_visible', is_visible, description)
            self.undo_stack.push(command)

    def on_layer_lock_changed(self, index: int, is_locked: bool):
        """[槽][已升级Undo] 当用户点击图层锁定图标时，创建命令。"""
        if self.current_char_obj and 0 <= index < len(self.current_char_obj.strokes):
            description = f"锁定图层 '笔画 {index + 1}'" if is_locked else f"解锁图层 '笔画 {index + 1}'"
            command = ModifyStrokePropertyCommand(self, index, 'is_locked', is_locked, description)
            self.undo_stack.push(command)
    
    def on_layer_opacity_set(self):
        """
        [槽函数][专业版] 当用户释放透明度滑块时调用，创建单个可撤销命令。

        此槽函数连接到 QSlider 的 sliderReleased 信号，确保只有在用户
        完成交互、确定最终值之后，才创建一个 ModifyStrokePropertyCommand。
        这避免了在拖动过程中产生大量冗余的撤销历史记录。
        """
        # --- 1. 获取当前选中的笔画对象 ---
        # 使用辅助方法来安全地获取，如果未选中任何图层，则直接返回
        stroke = self._get_selected_stroke()
        if not stroke:
            return

        # --- 2. 获取滑块的最终值并进行转换 ---
        # 从滑块获取 0-100 的整数值
        value = self.layer_opacity_slider.value()
        # 将其转换为 0.0-1.0 的浮点数，以匹配数据模型的要求
        new_opacity = value / 100.0
        
        # --- 3. 检查数值是否真的发生了变化 ---
        # 使用 math.isclose 进行浮点数比较，以避免因精度问题产生不必要的命令。
        # 只有当新旧值有实质性差异时，才继续执行。
        if not math.isclose(stroke.opacity, new_opacity, rel_tol=1e-3):
            
            # --- 4. 创建并推送撤销命令 ---
            # 获取当前选中项在模型中的真实索引
            index = self.layer_list_widget.currentItem().data(Qt.UserRole)
            
            # 构造一个清晰的描述，用于显示在“历史记录”和撤销菜单中
            description = f"设置图层 '{stroke.name}' 透明度为 {value}%"
            
            # 实例化我们通用的属性修改命令
            command = ModifyStrokePropertyCommand(
                main_window=self, 
                stroke_index=index, 
                prop_name='opacity', 
                new_value=new_opacity, 
                description=description
            )
            
            # 将命令推入撤销栈。这一步会自动执行 command.redo()，
            # 从而更新模型并刷新UI。
            self.undo_stack.push(command)

    def on_layer_rename_start(self, item: QListWidgetItem):
        """[槽] 当用户双击图层项时，准备进行重命名。"""
        widget = self.layer_list_widget.itemWidget(item)
        if isinstance(widget, LayerItemWidget):
            # 创建一个临时的 QLineEdit 用于编辑
            line_edit = QLineEdit(widget.name_label.text())
            
            # 当编辑完成时（按Enter或失去焦点），调用 on_layer_rename_finish
            line_edit.editingFinished.connect(lambda: self.on_layer_rename_finish(item, line_edit))
            
            # 用 QLineEdit 替换 QLabel
            widget.layout().replaceWidget(widget.name_label, line_edit)
            widget.name_label.hide()
            line_edit.setFocus()
            line_edit.selectAll()

    def on_layer_rename_finish(self, item: QListWidgetItem, line_edit: QLineEdit):
        """[槽][已升级Undo] 当图层重命名结束时，创建命令。"""
        widget = self.layer_list_widget.itemWidget(item)
        if isinstance(widget, LayerItemWidget):
            index = item.data(Qt.UserRole)
            stroke = self.current_char_obj.strokes[index]
            
            new_name = line_edit.text().strip()
            # 只有当名称有效且确实发生了改变时，才创建命令
            if new_name and new_name != stroke.name:
                description = f"重命名图层为 '{new_name}'"
                command = ModifyStrokePropertyCommand(self, index, 'name', new_name, description)
                self.undo_stack.push(command)
            
            # 无论是否创建命令，都需要恢复UI
            widget.layout().replaceWidget(line_edit, widget.name_label)
            line_edit.deleteLater()
            widget.name_label.show()
            # 更新一下列表，以防撤销/重做时名称不一致
            self.update_layer_list()
    def _update_views_after_property_change(self):
        """当笔画属性通过命令改变后，刷新必要的UI。"""
        self.canvas.update()
        self._update_layer_buttons_state() # 刷新透明度滑块等
        # 注意：这里不需要调用 _update_all_views()，因为它太“重”了，
        # 属性修改只需要刷新画布和图层面板本身。


# --- 撤销/重做命令类 ---
class AddStrokeCommand(QUndoCommand):
    def __init__(self, char_obj, stroke, main_window):
        super().__init__(f"添加笔画到 '{char_obj.char}'")
        self.char_obj, self.stroke, self.main = char_obj, stroke, main_window
    def redo(self):
        self.char_obj.add_stroke(self.stroke); self.main._update_all_views()
    def undo(self):
        self.char_obj._strokes.pop()
        self.char_obj.update_design_status()
        
        # [核心修复] 在刷新视图前，重置画布的选择状态
        if self.main.canvas:
            self.main.canvas.selected_stroke_index = -1
            self.main.canvas.selected_anchor_indices.clear()
            self.main.canvas.active_anchor_index = -1
        
        self.main._update_all_views()

class RemoveStrokeCommand(QUndoCommand):
    def __init__(self, char_obj, index, main_window):
        super().__init__(f"删除笔画 {index+1}")
        self.char_obj, self.index, self.main = char_obj, index, main_window
        self.stroke = self.char_obj.strokes[index]
    def redo(self):
        self.char_obj._strokes.pop(self.index)
        self.char_obj.update_design_status()

        # [核心修复] 在刷新视图前，重置画布的选择状态
        if self.main.canvas:
            self.main.canvas.selected_stroke_index = -1
            self.main.canvas.selected_anchor_indices.clear()
            self.main.canvas.active_anchor_index = -1

        self.main._update_all_views()
    def undo(self):
        self.char_obj._strokes.insert(self.index, self.stroke); self.char_obj.update_design_status(); self.main._update_all_views()
class ModifyStrokeCommand(QUndoCommand):
    """一个用于记录笔画修改操作的 QUndoCommand。"""
    def __init__(self, main_window: 'MainWindow', stroke_index: int, old_stroke: HandwritingStroke, new_stroke: HandwritingStroke):
        super().__init__(f"编辑笔画 {stroke_index + 1}")
        self.main = main_window
        self.stroke_index = stroke_index
        # 必须使用深拷贝来存储状态，防止后续修改影响记录
        self.old_stroke_state = old_stroke.copy()
        self.new_stroke_state = new_stroke.copy()

    def redo(self):
        """重做：应用新的笔画状态。"""
        char_obj = self.main.current_char_obj
        if char_obj and 0 <= self.stroke_index < len(char_obj.strokes):
            # 1. 更新数据模型
            char_obj._strokes[self.stroke_index] = self.new_stroke_state.copy()
            char_obj._mark_dirty()
            self.main.is_project_dirty = True

            # 2. [核心修复] 在刷新视图前，清除画布中过时的节点选择状态。
            #    因为路径的节点数量/索引已经改变，旧的选择不再有效。
            if self.main.canvas:
                self.main.canvas.selected_anchor_indices.clear()
                self.main.canvas.active_anchor_index = -1
            
            # 3. 刷新所有UI
            self.main._update_all_views()

    def undo(self):
        """撤销：恢复旧的笔画状态。"""
        char_obj = self.main.current_char_obj
        if char_obj and 0 <= self.stroke_index < len(char_obj.strokes):
            # 1. 恢复数据模型
            char_obj._strokes[self.stroke_index] = self.old_stroke_state.copy()
            char_obj._mark_dirty()
            self.main.is_project_dirty = True

            # 2. [核心修复] 撤销操作同样改变了路径拓扑，也需要清除节点选择。
            if self.main.canvas:
                self.main.canvas.selected_anchor_indices.clear()
                self.main.canvas.active_anchor_index = -1

            # 3. 刷新所有UI
            self.main._update_all_views()
class AddMultipleStrokesCommand(QUndoCommand):
    """一个用于一次性添加多个笔画的命令，确保单次操作的原子性。"""
    def __init__(self, char_obj: FontChar, strokes: List[HandwritingStroke], main_window: 'MainWindow'):
        super().__init__(f"矢量化字形 '{char_obj.char}'")
        self.char_obj = char_obj
        self.strokes_to_add = strokes
        self.main = main_window
        self.num_strokes = len(strokes)

    def redo(self):
        """重做：将所有笔画添加到字符中。"""
        for stroke in self.strokes_to_add:
            # 使用深拷贝确保每次操作的独立性
            self.char_obj.add_stroke(stroke.copy())
        self.main.is_project_dirty = True
        self.main._update_all_views()

    def undo(self):
        """撤销：从字符的末尾移除相应数量的笔画。"""
        if len(self.char_obj._strokes) >= self.num_strokes:
            self.char_obj._strokes = self.char_obj._strokes[:-self.num_strokes]
            self.char_obj.update_design_status()
            self.main.is_project_dirty = True
            self.main._update_all_views()
class ModifyStrokePropertyCommand(QUndoCommand):
    """
    [已增强] 一个通用的命令，用于修改单个笔画的单个属性，支持撤销/重做。
    新增了回调机制，以处理需要额外操作的属性修改（如路径缓存失效）。
    """
    def __init__(self, main_window: 'MainWindow', stroke_index: int, prop_name: str, new_value: Any, description: str):
        super().__init__(description)
        self.main = main_window
        self.char_obj = main_window.current_char_obj
        self.stroke_index = stroke_index
        self.prop_name = prop_name
        self.new_value = new_value

        stroke = self.char_obj.strokes[self.stroke_index]
        self.old_value = getattr(stroke, self.prop_name)
        
        self._invalidate_callback: Optional[Callable[[HandwritingStroke], None]] = None

    def set_invalidate_callback(self, callback: Callable[[HandwritingStroke], None]):
        """
        [新增] 设置一个在 redo/undo 后调用的回调函数。
        用于处理需要使缓存失效等额外操作的属性。
        """
        self._invalidate_callback = callback

    def redo(self):
        """重做：将新值应用于属性。"""
        if self.char_obj and 0 <= self.stroke_index < len(self.char_obj.strokes):
            stroke = self.char_obj.strokes[self.stroke_index]
            setattr(stroke, self.prop_name, self.new_value)
            
            # 如果设置了回调，则执行它
            if self._invalidate_callback:
                self._invalidate_callback(stroke)
            
            self.main.is_project_dirty = True
            self.main._update_views_after_property_change()

    def undo(self):
        """撤销：将旧值恢复给属性。"""
        if self.char_obj and 0 <= self.stroke_index < len(self.char_obj.strokes):
            stroke = self.char_obj.strokes[self.stroke_index]
            setattr(stroke, self.prop_name, self.old_value)

            # 如果设置了回调，则执行它
            if self._invalidate_callback:
                self._invalidate_callback(stroke)

            self.main.is_project_dirty = True
            self.main._update_views_after_property_change()



# ==============================================================================
# SECTION 5: 应用程序入口点 (APPLICATION ENTRY POINT)
#
# 这是 Python 脚本的主入口。当直接运行此文件时，这里的代码将被执行。
# ==============================================================================

if __name__ == '__main__':
    # --- 步骤 1: 设置应用程序级别的属性 ---

    # 启用高 DPI (High Dots Per Inch) 缩放支持。
    # 这是确保应用程序在现代高分辨率显示器（如 4K 屏或 Retina 屏）上
    # 看起来清晰、不模糊的关键设置。
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # --- 步骤 2: 创建 QApplication 实例 ---
    # 任何一个 PyQt5 应用程序都必须有且只有一个 QApplication 实例。
    # sys.argv 允许从命令行传递参数给应用程序。
    app = QApplication(sys.argv)
    
    # --- 步骤 3: 设置应用程序的全局样式 ---
    # "Fusion" 是一个现代、简洁且跨平台的样式，推荐在所有系统上使用
    # 以获得一致的外观。其他选项有 "Windows", "WindowsVista", "GTK+" 等。
    app.setStyle("Fusion")

    # --- 步骤 4: 实例化并显示主窗口 ---
    # 创建我们之前定义的 MainWindow 类的实例。
    # 这将触发 __init__ 方法，从而构建整个UI和加载数据。
    main_window = MainWindow()
    
    # 调用 .show() 方法来让主窗口在屏幕上可见。
    main_window.show()
    
    # --- 步骤 5: 启动事件循环并处理退出 ---
    # app.exec_() 启动了 Qt 的事件循环。
    # 程序将在此处暂停，等待并处理用户的交互（如点击、键盘输入等），
    # 直到最后一个窗口被关闭。
    # sys.exit() 确保应用程序在退出时能返回一个正确的状态码给操作系统。
    sys.exit(app.exec_())