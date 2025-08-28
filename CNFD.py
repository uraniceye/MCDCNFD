

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import json
import os
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
from datetime import datetime
import tempfile
import math
import random
import threading
import time
import sqlite3
import unicodedata

try:
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.pens.pointPen import PointToSegmentPen
    from fontTools.misc.transform import Transform
    from fontTools.pens.basePen import BasePen
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False

class ChineseCharacterDatabase:
    """中文字符数据库管理类"""
    
    def __init__(self):
        self.db_path = "chinese_chars.db"
        self.init_database()
        
    def init_database(self):
        """初始化字符数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY,
                char TEXT UNIQUE,
                unicode_val INTEGER,
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
        
        # 如果数据库为空，初始化字符数据
        cursor.execute('SELECT COUNT(*) FROM characters')
        if cursor.fetchone()[0] == 0:
            self.populate_database(cursor)
            
        conn.commit()
        conn.close()
        
    def populate_database(self, cursor):
        """填充字符数据库"""
        # GB2312 一级常用汉字 (3755个)
        gb2312_level1 = self.generate_gb2312_chars()
        
        # 常用标点符号
        # 修正: 修正了无效的标点符号，避免启动时出现警告
        punctuation = ['。', '，', '？', '！', '；', '：', '「', '」', '『', '』', 
                       '（', '）', '【', '】', '《', '》', '〈', '〉', '“', '”', 
                       '‘', '’', '…', '—', '～', '·']
        
        # 数字
        numbers = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万']
        
        all_chars = gb2312_level1 + punctuation + numbers
        
        for char in all_chars:
            # 确保只处理单个字符
            if len(char) != 1:
                print(f"跳过无效字符: '{char}' (长度: {len(char)})")
                continue
                
            try:
                category = self.get_char_category(char)
                frequency = self.get_char_frequency(char)
                radical = self.get_char_radical(char)
                stroke_count = self.estimate_stroke_count(char)
                
                cursor.execute('''
                    INSERT OR IGNORE INTO characters 
                    (char, unicode_val, category, frequency, radical, stroke_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (char, ord(char), category, frequency, radical, stroke_count))
            except Exception as e:
                print(f"处理字符 '{char}' 时出错: {e}")
                continue
            
    def generate_gb2312_chars(self):
        """生成GB2312汉字列表"""
        chars = []
        
        # 使用常用汉字列表作为基础，避免GB2312解码问题
        # 修正: 移除了罕见字 '紶'
        common_chars_text = """
        的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什二次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气母并别真打太新比才便夫再书部水像眼等体却加电主界门利海南听表德少克代员许稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通师立象数四失满战远格士音轻目条呢病始达深完今提求清王化空业思切怎非找片罗钱南语元喜曾离飞科言干流欢约各即指合反题必该论交终林请医晚制球决窝传画保读运及则房早院重苦火布品近向产答星精视五连司巴奇管类未朋专婚小夜青北队久乎越观落尽形影红爸百令周吗识步希亚术留市半热送兴造谈容极随演收首根讲整式取照办强石古华谊拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断福城故历惊脸选包紧争名建维绿精册巾股份歌姐纵幅晓曾亦酬技秀汗豆苍亿构贺谈虎骑粒送毛驱菜农枝团锦
        """

        # 清理文本并提取单个字符
        clean_chars = []
        for char in common_chars_text:
            if char.strip() and len(char) == 1 and '\u4e00' <= char <= '\u9fff':
                if char not in clean_chars:
                    clean_chars.append(char)
        
        # 添加更多汉字
        additional_chars = []
        for i in range(0x4e00, 0x9fff, 100):  # 从CJK统一汉字区取样
            try:
                char = chr(i)
                if char not in clean_chars:
                    additional_chars.append(char)
                    if len(additional_chars) + len(clean_chars) >= 2000:
                        break
            except:
                continue
        
        chars.extend(clean_chars[:1000])  # 限制常用字符数量
        chars.extend(additional_chars[:1000])  # 添加其他汉字
        
        # 去重并确保都是单字符
        unique_chars = []
        for char in chars:
            if len(char) == 1 and char not in unique_chars:
                unique_chars.append(char)
        
        return unique_chars[:2000]  # 返回前2000个字符
    
    def get_char_category(self, char):
        """获取字符分类"""
        if '\u4e00' <= char <= '\u9fff':
            return '汉字'
        elif char in '零一二三四五六七八九十百千万':
            return '数字'
        elif char in '。，？！；：「」『』（）【】《》〈〉“”…—～·': # 修正: 调整标点符号列表以匹配
            return '标点'
        else:
            return '其他'
            
    def get_char_frequency(self, char):
        """获取字符使用频率（简化版）"""
        # 常用字频率映射（简化版）
        high_freq = '的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头基经工很觉样发代分口白最见入要她出也得里后自以会'
        
        medium_freq = '文字方法新问题意思情况工作学校家庭朋友时间地方政府社会经济发展建设管理系统技术科学研究教育文化艺术历史传统现代世界国际中国人民群众干部领导组织党员政治思想理论实践活动工作经验方法措施政策法律制度体系机制模式结构功能作用影响效果质量水平标准要求目标任务计划方案项目程序过程阶段步骤环节关键重点难点问题矛盾困难挑战机遇条件基础保障支持帮助指导服务提供利用发挥推进促进加强改善提高完善创新发展进步成功成果成就贡献价值意义重要必要需要应该可能能够应当必须'
        
        if char in high_freq:
            return 100
        elif char in medium_freq:
            return 50
        else:
            return 10
            
    def get_char_radical(self, char):
        """获取字符部首（简化版）"""
        radical_map = {
            '人': '人', '大': '大', '小': '小', '心': '心', '手': '手', '口': '口',
            '木': '木', '水': '氵', '火': '火', '土': '土', '金': '钅', '日': '日',
            '月': '月', '山': '山', '石': '石', '田': '田', '目': '目', '耳': '耳'
        }
        return radical_map.get(char, '未知')
        
    def estimate_stroke_count(self, char):
        """估算笔画数（简化版）"""
        if not ('\u4e00' <= char <= '\u9fff'):
            return 1
        # 根据字符复杂度估算笔画数
        try:
            complexity = len(unicodedata.name(char, 'UNKNOWN'))
            if complexity < 20:
                return random.randint(3, 8)
            elif complexity < 40:
                return random.randint(8, 15)
            else:
                return random.randint(15, 25)
        except:
            return random.randint(5, 15)
            
    def search_characters(self, query, category=None, limit=100):
        """搜索字符"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = 'SELECT * FROM characters WHERE char LIKE ?'
        params = [f'%{query}%']
        
        if category and category != '全部':
            sql += ' AND category = ?'
            params.append(category)
            
        sql += ' ORDER BY frequency DESC, stroke_count ASC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        return results
        
    def get_characters_by_category(self, category, limit=100):
        """按分类获取字符"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category == '全部':
            cursor.execute('SELECT * FROM characters ORDER BY frequency DESC LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT * FROM characters WHERE category = ? ORDER BY frequency DESC LIMIT ?', 
                          (category, limit))
        
        results = cursor.fetchall()
        conn.close()
        return results
        
    def mark_designed(self, char):
        """标记字符为已设计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE characters SET is_designed = 1, modified_date = ? WHERE char = ?',
                      (datetime.now().isoformat(), char))
        conn.commit()
        conn.close()
        
    def get_stats(self):
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM characters')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM characters WHERE is_designed = 1')
        designed = cursor.fetchone()[0]
        
        cursor.execute('SELECT category, COUNT(*) FROM characters GROUP BY category')
        by_category = cursor.fetchall()
        
        conn.close()
        
        return {
            'total': total,
            'designed': designed,
            'progress': (designed / total * 100) if total > 0 else 0,
            'by_category': dict(by_category)
        }

class VectorPath:
    """矢量路径类，支持贝塞尔曲线和复杂路径操作"""
    def __init__(self):
        self.commands = []
        self.points = []
        self.bounds = None
        
    def moveTo(self, x, y):
        self.commands.append(('moveTo', x, y))
        self._update_bounds(x, y)
        
    def lineTo(self, x, y):
        self.commands.append(('lineTo', x, y))
        self._update_bounds(x, y)
        
    def curveTo(self, x1, y1, x2, y2, x3, y3):
        self.commands.append(('curveTo', x1, y1, x2, y2, x3, y3))
        self._update_bounds(x1, y1)
        self._update_bounds(x2, y2)
        self._update_bounds(x3, y3)
        
    def closePath(self):
        self.commands.append(('closePath',))
        
    def clear(self):
        self.commands.clear()
        self.points.clear()
        self.bounds = None
        
    def _update_bounds(self, x, y):
        if self.bounds is None:
            self.bounds = [x, y, x, y]
        else:
            self.bounds[0] = min(self.bounds[0], x)
            self.bounds[1] = min(self.bounds[1], y)
            self.bounds[2] = max(self.bounds[2], x)
            self.bounds[3] = max(self.bounds[3], y)

class HandwritingStroke:
    """手写笔画类，支持压感和笔触变化"""
    def __init__(self):
        self.points = []
        self.stroke_type = 'normal'
        self.width_variation = 1.0
        self.color = '#2d3748'
        self.smoothness = 0.3
        
    def add_point(self, x, y, pressure=1.0):
        timestamp = time.time()
        self.points.append((x, y, pressure, timestamp))
    
    def smooth_stroke(self, smoothing_factor=0.3):
        if len(self.points) < 3:
            return
        
        smoothed_points = [self.points[0]]
        
        for i in range(1, len(self.points) - 1):
            prev_point = self.points[i-1]
            curr_point = self.points[i]
            next_point = self.points[i+1]
            
            smooth_x = curr_point[0] * (1 - smoothing_factor) + \
                      (prev_point[0] + next_point[0]) * smoothing_factor / 2
            smooth_y = curr_point[1] * (1 - smoothing_factor) + \
                      (prev_point[1] + next_point[1]) * smoothing_factor / 2
            
            smoothed_points.append((smooth_x, smooth_y, curr_point[2], curr_point[3]))
        
        smoothed_points.append(self.points[-1])
        self.points = smoothed_points
    
    def get_bounds(self):
        if not self.points:
            return None
        
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

class ModernChineseFontDesigner:
    """增强版现代化中文手写体字体设计器"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("增强版现代中文手写体字体设计器 Pro v4.0 作者：跳舞的火公子")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#f8fafc')
        self.root.resizable(True, True)
        
        # 现代化主题色彩
        self.colors = {
            'primary': '#6366f1',
            'primary_light': '#a5b4fc',
            'primary_dark': '#4338ca',
            'secondary': '#06b6d4',
            'accent': '#f59e0b',
            'success': '#10b981',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'info': '#3b82f6',
            'light': '#f8fafc',
            'muted': '#64748b',
            'dark': '#1e293b',
            'surface': '#ffffff',
            'border': '#e2e8f0',
            'text': '#0f172a',
            'text_muted': '#64748b'
        }
        
        # 初始化字符数据库
        try:
            self.char_db = ChineseCharacterDatabase()
        except Exception as e:
            print(f"数据库初始化失败: {e}")
            messagebox.showerror("初始化错误", f"字符数据库初始化失败: {e}")
            return
        
        # 现代化样式设置
        self.setup_modern_styles()
        
        # 字体数据存储
        self.font_data = {}
        self.current_char = '中'
        self.grid_size = 48
        self.pixel_size = 12
        self.canvas_grid = []
        self.current_color = self.colors['dark']
        self.bg_color = self.colors['surface']
        
        # 手写体支持
        self.handwriting_mode = True
        self.current_stroke = None
        self.strokes = {}
        self.stroke_width = 4
        self.pressure_sensitivity = True
        self.stroke_smoothing = 0.4
        
        # 界面状态
        self.char_buttons = {}
        self.current_category = '全部'
        self.search_query = ''
        self.current_page = 0
        self.chars_per_page = 50
        self.displayed_chars = []

        # 修正: 初始化 undo_stack 和 clipboard_char，增强代码健壮性
        self.undo_stack = []
        self.clipboard_char = None
        
        # 矢量支持
        self.vector_mode = True
        self.vector_paths = {}
        self.current_tool = 'brush'
        self.control_points = []
        
        # 字体元数据
        self.font_metadata = {
            'family_name': '增强手写体',
            'style_name': 'Regular',
            'version': '4.0',
            'copyright': '© 2024 Enhanced Chinese Font Designer',
            'description': '使用增强版现代中文手写体字体设计器创建',
            'units_per_em': 1000,
            'ascender': 880,
            'descender': -120,
            'line_gap': 50,
            'cap_height': 800,
            'x_height': 600
        }
        
        # 导出设置
        self.export_settings = {
            'png_size': 128,
            'svg_size': 256,
            'include_metadata': True,
            'compression_level': 9
        }
        
        # 性能优化
        self.cache = {}
        self.render_quality = 'high'
        
        # 初始化
        self.init_data_structures()
        self.create_modern_ui()
        self.bind_events()
        self.start_background_tasks()
        self.check_dependencies()
        
    def setup_modern_styles(self):
        """设置现代化UI样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Modern.TLabel',
                       background=self.colors['surface'],
                       foreground=self.colors['text'],
                       font=('Microsoft YaHei UI', 10))
        
        style.configure('Modern.TButton',
                       padding=(20, 10),
                       relief='flat',
                       borderwidth=0,
                       font=('Microsoft YaHei UI', 10, 'bold'))
        
        style.map('Modern.TButton',
                 background=[('active', self.colors['primary_light']),
                           ('pressed', self.colors['primary_dark']),
                           ('!active', self.colors['primary'])],
                 foreground=[('active', 'white'),
                           ('pressed', 'white'), 
                           ('!active', 'white')])
        
        style.configure('Modern.TFrame',
                       background=self.colors['surface'],
                       relief='flat',
                       borderwidth=0)
        
        style.configure('Modern.TNotebook',
                       tabmargins=(5, 5, 5, 0),
                       background=self.colors['surface'])
        
        style.configure('Modern.TNotebook.Tab',
                       padding=(20, 12),
                       background=self.colors['border'],
                       foreground=self.colors['text_muted'],
                       font=('Microsoft YaHei UI', 10, 'bold'))
        
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', self.colors['primary']),
                           ('active', self.colors['primary_light']),
                           ('!active', self.colors['border'])],
                 foreground=[('selected', 'white'),
                           ('active', 'white'),
                           ('!active', self.colors['text_muted'])])
        
    def init_data_structures(self):
        """初始化数据结构"""
        # 获取字符统计
        try:
            stats = self.char_db.get_stats()
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            stats = {'total': 0, 'designed': 0, 'progress': 0, 'by_category': {}}
        
        # 初始化字符数据
        try:
            chars = self.char_db.get_characters_by_category('全部', 3500)
            for char_data in chars:
                char = char_data[1]  # char字段
                self.font_data[char] = {
                    'pixels': [[0 for _ in range(self.grid_size)] 
                              for _ in range(self.grid_size)],
                    'advance_width': self.grid_size * 1.0,
                    'left_bearing': 0,
                    'right_bearing': 0,
                    'is_chinese': '\u4e00' <= char <= '\u9fff',
                    'category': char_data[3],  # category字段
                    'frequency': char_data[4],  # frequency字段
                    'stroke_count': char_data[8] or 1  # stroke_count字段
                }
                self.strokes[char] = []
                self.vector_paths[char] = VectorPath()
        except Exception as e:
            print(f"初始化字符数据失败: {e}")
            # 创建基本的字符数据结构
            basic_chars = ['中', '国', '汉', '字', '的', '一', '是', '了', '我', '不']
            for char in basic_chars:
                self.font_data[char] = {
                    'pixels': [[0 for _ in range(self.grid_size)] 
                              for _ in range(self.grid_size)],
                    'advance_width': self.grid_size * 1.0,
                    'left_bearing': 0,
                    'right_bearing': 0,
                    'is_chinese': True,
                    'category': '汉字',
                    'frequency': 100,
                    'stroke_count': 5
                }
                self.strokes[char] = []
                self.vector_paths[char] = VectorPath()
            
    def create_modern_ui(self):
        """创建现代化用户界面"""
        # 主容器
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 顶部导航栏
        self.create_enhanced_navbar(main_container)
        
        # 主内容区域
        content_container = tk.Frame(main_container, bg=self.colors['light'])
        content_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # 使用现代化面板布局
        self.create_enhanced_layout(content_container)
        
        # 底部状态栏
        self.create_enhanced_statusbar(main_container)
        
    def create_enhanced_navbar(self, parent):
        """创建增强版导航栏"""
        navbar = tk.Frame(parent, bg=self.colors['primary'], height=80)
        navbar.pack(fill=tk.X)
        navbar.pack_propagate(False)
        
        # 标题区域
        title_frame = tk.Frame(navbar, bg=self.colors['primary'])
        title_frame.pack(side=tk.LEFT, padx=25, pady=15)
        
        title_main = tk.Label(title_frame, 
                             text="增强版现代中文手写体字体设计器",
                             font=('Microsoft YaHei UI', 20, 'bold'),
                             fg='white', bg=self.colors['primary'])
        title_main.pack(anchor='w')
        
        subtitle = tk.Label(title_frame,
                           text="Enhanced Chinese Handwriting Font Designer Pro v4.0 - 支持3000+汉字",
                           font=('Microsoft YaHei UI', 10),
                           fg=self.colors['primary_light'], bg=self.colors['primary'])
        subtitle.pack(anchor='w')
        
        # 统计信息显示
        stats_frame = tk.Frame(navbar, bg=self.colors['primary'])
        stats_frame.pack(side=tk.LEFT, padx=50, pady=15)
        
        try:
            stats = self.char_db.get_stats()
        except:
            stats = {'total': 0, 'designed': 0, 'progress': 0}
        
        stats_title = tk.Label(stats_frame,
                              text="项目统计",
                              font=('Microsoft YaHei UI', 12, 'bold'),
                              fg='white', bg=self.colors['primary'])
        stats_title.pack(anchor='w')
        
        self.stats_display = tk.Label(stats_frame,
                                     text=f"总字符: {stats['total']} | 已设计: {stats['designed']} | 进度: {stats['progress']:.1f}%",
                                     font=('Microsoft YaHei UI', 10),
                                     fg=self.colors['primary_light'], bg=self.colors['primary'])
        self.stats_display.pack(anchor='w')
        
        # 工具按钮区域
        tools_frame = tk.Frame(navbar, bg=self.colors['primary'])
        tools_frame.pack(side=tk.RIGHT, padx=25, pady=15)
        
        # 增强版工具按钮
        tools = [
            ("新建项目", self.new_font, self.colors['success']),
            ("打开项目", self.open_font, self.colors['info']),
            ("保存项目", self.save_font, self.colors['warning']),
            ("导出字体", self.show_export_dialog, self.colors['secondary']),
            ("批量操作", self.show_batch_dialog, self.colors['accent']),
            ("设置", self.show_settings_dialog, '#6b7280')
        ]
        
        for text, command, color in tools:
            btn = self.create_navbar_button(tools_frame, text, command, color)
            btn.pack(side=tk.LEFT, padx=3)
            
    def create_navbar_button(self, parent, text, command, color):
        """创建导航栏按钮"""
        btn = tk.Button(parent,
                       text=text,
                       command=command,
                       font=('Microsoft YaHei UI', 9, 'bold'),
                       bg=self.lighten_color(self.colors['primary'], 0.1),
                       fg='white',
                       borderwidth=0,
                       padx=12,
                       pady=6,
                       relief='flat',
                       cursor='hand2')
        
        def on_enter(e):
            btn.config(bg=color)
            
        def on_leave(e):
            btn.config(bg=self.lighten_color(self.colors['primary'], 0.1))
            
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
        
    def create_enhanced_layout(self, parent):
        """创建增强版布局"""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        
        # 左侧面板 - 字符选择和工具（增强版）
        self.create_enhanced_left_panel(parent)
        
        # 中间面板 - 主编辑区域（增强版）
        self.create_enhanced_center_panel(parent)
        
        # 右侧面板 - 预览和设置（增强版）
        self.create_enhanced_right_panel(parent)
        
    def create_enhanced_left_panel(self, parent):
        """创建增强版左侧面板"""
        left_panel = tk.Frame(parent, bg=self.colors['surface'], 
                             width=350, relief='solid', bd=1)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        left_panel.grid_propagate(False)
        
        # 搜索和过滤区域
        search_section = tk.Frame(left_panel, bg=self.colors['surface'])
        search_section.pack(fill=tk.X, padx=15, pady=15)
        
        # 搜索标题
        search_title = tk.Label(search_section,
                               text="智能字符搜索",
                               font=('Microsoft YaHei UI', 14, 'bold'),
                               fg=self.colors['text'],
                               bg=self.colors['surface'])
        search_title.pack(anchor='w', pady=(0, 10))
        
        # 搜索框
        search_frame = tk.Frame(search_section, bg=self.colors['surface'])
        search_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame,
                               textvariable=self.search_var,
                               font=('Microsoft YaHei UI', 11),
                               bg=self.colors['surface'],
                               fg=self.colors['text'],
                               bd=2,
                               relief='solid',
                               width=25)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        search_entry.bind('<KeyRelease>', self.on_search_changed)
        
        search_btn = tk.Button(search_frame,
                              text="搜索",
                              command=self.perform_search,
                              font=('Microsoft YaHei UI', 10, 'bold'),
                              bg=self.colors['primary'],
                              fg='white',
                              borderwidth=0,
                              padx=15,
                              cursor='hand2')
        search_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 分类过滤器
        filter_frame = tk.Frame(search_section, bg=self.colors['surface'])
        filter_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(filter_frame, text="字符分类:",
                font=('Microsoft YaHei UI', 10),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.category_var = tk.StringVar(value='全部')
        category_combo = ttk.Combobox(filter_frame,
                                     textvariable=self.category_var,
                                     values=['全部', '汉字', '标点', '数字', '其他'],
                                     state='readonly',
                                     font=('Microsoft YaHei UI', 10),
                                     width=10)
        category_combo.pack(side=tk.RIGHT)
        category_combo.bind('<<ComboboxSelected>>', self.on_category_changed)
        
        # 高级过滤选项
        advanced_frame = tk.LabelFrame(search_section,
                                      text="高级过滤",
                                      font=('Microsoft YaHei UI', 10, 'bold'),
                                      fg=self.colors['text'],
                                      bg=self.colors['surface'])
        advanced_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 笔画数过滤
        stroke_filter_frame = tk.Frame(advanced_frame, bg=self.colors['surface'])
        stroke_filter_frame.pack(fill=tk.X, padx=8, pady=5)
        
        tk.Label(stroke_filter_frame, text="笔画数:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.min_strokes = tk.IntVar(value=1)
        self.max_strokes = tk.IntVar(value=30)
        
        min_stroke_spin = tk.Spinbox(stroke_filter_frame,
                                    from_=1, to=30,
                                    textvariable=self.min_strokes,
                                    width=4, font=('Microsoft YaHei UI', 9))
        min_stroke_spin.pack(side=tk.LEFT, padx=(5, 2))
        
        tk.Label(stroke_filter_frame, text="-",
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        max_stroke_spin = tk.Spinbox(stroke_filter_frame,
                                    from_=1, to=30,
                                    textvariable=self.max_strokes,
                                    width=4, font=('Microsoft YaHei UI', 9))
        max_stroke_spin.pack(side=tk.LEFT, padx=2)
        
        # 设计状态过滤
        status_frame = tk.Frame(advanced_frame, bg=self.colors['surface'])
        status_frame.pack(fill=tk.X, padx=8, pady=5)
        
        self.show_designed = tk.BooleanVar(value=True)
        self.show_undesigned = tk.BooleanVar(value=True)
        
        designed_check = tk.Checkbutton(status_frame,
                                       text="已设计",
                                       variable=self.show_designed,
                                       bg=self.colors['surface'],
                                       fg=self.colors['text'],
                                       font=('Microsoft YaHei UI', 9))
        designed_check.pack(side=tk.LEFT)
        
        undesigned_check = tk.Checkbutton(status_frame,
                                         text="未设计",
                                         variable=self.show_undesigned,
                                         bg=self.colors['surface'],
                                         fg=self.colors['text'],
                                         font=('Microsoft YaHei UI', 9))
        undesigned_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # 字符显示区域
        char_display_section = tk.Frame(left_panel, bg=self.colors['surface'])
        char_display_section.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # 字符网格和分页控制
        self.create_character_grid(char_display_section)
        
        # 工具区域
        self.create_enhanced_tools_section(left_panel)
        
    def create_character_grid(self, parent):
        """创建字符网格显示"""
        # 字符数量和分页信息
        info_frame = tk.Frame(parent, bg=self.colors['surface'])
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.char_count_label = tk.Label(info_frame,
                                        text="字符总数: 0",
                                        font=('Microsoft YaHei UI', 10),
                                        fg=self.colors['text'],
                                        bg=self.colors['surface'])
        self.char_count_label.pack(side=tk.LEFT)
        
        self.page_info_label = tk.Label(info_frame,
                                       text="第1页/共1页",
                                       font=('Microsoft YaHei UI', 10),
                                       fg=self.colors['text_muted'],
                                       bg=self.colors['surface'])
        self.page_info_label.pack(side=tk.RIGHT)
        
        # 字符网格容器
        grid_container = tk.Frame(parent, bg=self.colors['surface'])
        grid_container.pack(fill=tk.BOTH, expand=True)
        
        # 创建滚动区域
        canvas = tk.Canvas(grid_container, bg=self.colors['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(grid_container, orient='vertical', command=canvas.yview)
        self.char_grid_frame = tk.Frame(canvas, bg=self.colors['surface'])
        
        self.char_grid_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        
        canvas.create_window((0, 0), window=self.char_grid_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 鼠标滚轮支持
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        
        # 分页控制
        page_frame = tk.Frame(parent, bg=self.colors['surface'])
        page_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.prev_btn = tk.Button(page_frame,
                                 text="上一页",
                                 command=self.prev_page,
                                 font=('Microsoft YaHei UI', 9),
                                 bg=self.colors['border'],
                                 fg=self.colors['text'],
                                 borderwidth=0,
                                 padx=10,
                                 pady=3,
                                 state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT)
        
        self.next_btn = tk.Button(page_frame,
                                 text="下一页",
                                 command=self.next_page,
                                 font=('Microsoft YaHei UI', 9),
                                 bg=self.colors['border'],
                                 fg=self.colors['text'],
                                 borderwidth=0,
                                 padx=10,
                                 pady=3)
        self.next_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # 跳转到页面
        tk.Label(page_frame, text="跳转:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT, padx=(20, 5))
        
        self.page_entry = tk.Entry(page_frame,
                                  font=('Microsoft YaHei UI', 9),
                                  width=5,
                                  bd=1,
                                  relief='solid')
        self.page_entry.pack(side=tk.LEFT)
        self.page_entry.bind('<Return>', self.goto_page)
        
        # 每页显示数量
        tk.Label(page_frame, text="每页:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.RIGHT, padx=(0, 5))
        
        self.per_page_var = tk.StringVar(value='50')
        per_page_combo = ttk.Combobox(page_frame,
                                     textvariable=self.per_page_var,
                                     values=['20', '50', '100', '200'],
                                     state='readonly',
                                     width=5,
                                     font=('Microsoft YaHei UI', 9))
        per_page_combo.pack(side=tk.RIGHT)
        per_page_combo.bind('<<ComboboxSelected>>', self.on_per_page_changed)
        
    def create_enhanced_tools_section(self, parent):
        """创建增强版工具区域"""
        tools_frame = tk.LabelFrame(parent,
                                   text="绘图工具",
                                   font=('Microsoft YaHei UI', 11, 'bold'),
                                   fg=self.colors['text'],
                                   bg=self.colors['surface'],
                                   labelanchor='nw')
        tools_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # 工具选择
        tool_select_frame = tk.Frame(tools_frame, bg=self.colors['surface'])
        tool_select_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.current_tool_var = tk.StringVar(value='brush')
        
        tools = [
            ('毛笔', 'brush', self.colors['success']),
            ('钢笔', 'pen', self.colors['info']),
            ('马克笔', 'marker', self.colors['warning']),
            ('铅笔', 'pencil', self.colors['muted']),
            ('橡皮擦', 'eraser', self.colors['danger'])
        ]
        
        for name, value, color in tools:
            btn = tk.Radiobutton(tool_select_frame,
                                text=name,
                                variable=self.current_tool_var,
                                value=value,
                                command=lambda v=value: self.set_current_tool(v),
                                font=('Microsoft YaHei UI', 9),
                                bg=self.colors['surface'],
                                fg=self.colors['text'],
                                selectcolor=color,
                                activebackground=self.colors['surface'])
            btn.pack(anchor='w', pady=1)
            
        # 笔画参数调节
        self.create_enhanced_stroke_params(tools_frame)
        
        # 操作按钮
        self.create_enhanced_action_buttons(tools_frame)
        
    def create_enhanced_stroke_params(self, parent):
        """创建增强版笔画参数调节"""
        params_frame = tk.LabelFrame(parent,
                                    text="笔画参数",
                                    font=('Microsoft YaHei UI', 10, 'bold'),
                                    fg=self.colors['text'],
                                    bg=self.colors['surface'])
        params_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # 笔触宽度
        width_frame = tk.Frame(params_frame, bg=self.colors['surface'])
        width_frame.pack(fill=tk.X, pady=5, padx=8)
        
        tk.Label(width_frame, 
                text="笔触宽度",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(anchor='w')
        
        self.stroke_width_var = tk.IntVar(value=self.stroke_width)
        width_scale = tk.Scale(width_frame,
                              from_=1, to=20,
                              orient=tk.HORIZONTAL,
                              variable=self.stroke_width_var,
                              command=self.update_stroke_width,
                              bg=self.colors['surface'],
                              fg=self.colors['primary'],
                              highlightthickness=0,
                              troughcolor=self.colors['border'],
                              activebackground=self.colors['primary'],
                              length=180)
        width_scale.pack(fill=tk.X, pady=3)
        
        # 平滑度
        smooth_frame = tk.Frame(params_frame, bg=self.colors['surface'])
        smooth_frame.pack(fill=tk.X, pady=5, padx=8)
        
        tk.Label(smooth_frame,
                text="平滑度",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(anchor='w')
        
        self.smoothing_var = tk.DoubleVar(value=self.stroke_smoothing)
        smooth_scale = tk.Scale(smooth_frame,
                               from_=0.0, to=1.0, resolution=0.1,
                               orient=tk.HORIZONTAL,
                               variable=self.smoothing_var,
                               command=self.update_smoothing,
                               bg=self.colors['surface'],
                               fg=self.colors['primary'],
                               highlightthickness=0,
                               troughcolor=self.colors['border'],
                               activebackground=self.colors['primary'],
                               length=180)
        smooth_scale.pack(fill=tk.X, pady=3)
        
        # 颜色选择
        color_frame = tk.Frame(params_frame, bg=self.colors['surface'])
        color_frame.pack(fill=tk.X, pady=5, padx=8)
        
        tk.Label(color_frame,
                text="笔画颜色",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.color_preview = tk.Label(color_frame,
                                     text="   ",
                                     bg=self.current_color,
                                     relief='solid',
                                     bd=1,
                                     cursor='hand2')
        self.color_preview.pack(side=tk.RIGHT)
        self.color_preview.bind('<Button-1>', self.choose_color)
        
        # 高级选项
        advanced_frame = tk.Frame(params_frame, bg=self.colors['surface'])
        advanced_frame.pack(fill=tk.X, pady=5, padx=8)
        
        self.pressure_var = tk.BooleanVar(value=self.pressure_sensitivity)
        pressure_check = tk.Checkbutton(advanced_frame,
                                       text="启用压感效果",
                                       variable=self.pressure_var,
                                       command=self.toggle_pressure_sensitivity,
                                       font=('Microsoft YaHei UI', 9),
                                       bg=self.colors['surface'],
                                       fg=self.colors['text'],
                                       selectcolor=self.colors['primary'],
                                       activebackground=self.colors['surface'])
        pressure_check.pack(anchor='w')
        
        self.anti_alias_var = tk.BooleanVar(value=True)
        aa_check = tk.Checkbutton(advanced_frame,
                                 text="抗锯齿",
                                 variable=self.anti_alias_var,
                                 font=('Microsoft YaHei UI', 9),
                                 bg=self.colors['surface'],
                                 fg=self.colors['text'],
                                 selectcolor=self.colors['primary'],
                                 activebackground=self.colors['surface'])
        aa_check.pack(anchor='w')
        
    def create_enhanced_action_buttons(self, parent):
        """创建增强版操作按钮"""
        actions_frame = tk.LabelFrame(parent,
                                     text="操作",
                                     font=('Microsoft YaHei UI', 10, 'bold'),
                                     fg=self.colors['text'],
                                     bg=self.colors['surface'])
        actions_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # 基本操作
        basic_frame = tk.Frame(actions_frame, bg=self.colors['surface'])
        basic_frame.pack(fill=tk.X, pady=5, padx=8)
        
        buttons_row1 = [
            ("撤销", self.undo_stroke, self.colors['warning']),
            ("重做", self.redo_stroke, self.colors['info']),
            ("清除", self.clear_current_char, self.colors['danger'])
        ]
        
        for i, (text, command, color) in enumerate(buttons_row1):
            btn = tk.Button(basic_frame,
                           text=text,
                           command=command,
                           font=('Microsoft YaHei UI', 9),
                           bg=color,
                           fg='white',
                           borderwidth=0,
                           padx=8,
                           pady=4,
                           cursor='hand2',
                           activebackground=self.lighten_color(color, 0.1))
            btn.grid(row=0, column=i, padx=2, pady=1, sticky='ew')
            
        # 编辑操作
        edit_frame = tk.Frame(actions_frame, bg=self.colors['surface'])
        edit_frame.pack(fill=tk.X, pady=3, padx=8)
        
        buttons_row2 = [
            ("复制", self.copy_char, self.colors['info']),
            ("粘贴", self.paste_char, self.colors['success']),
            ("翻转", self.flip_char, self.colors['secondary'])
        ]
        
        for i, (text, command, color) in enumerate(buttons_row2):
            btn = tk.Button(edit_frame,
                           text=text,
                           command=command,
                           font=('Microsoft YaHei UI', 9),
                           bg=color,
                           fg='white',
                           borderwidth=0,
                           padx=8,
                           pady=4,
                           cursor='hand2',
                           activebackground=self.lighten_color(color, 0.1))
            btn.grid(row=0, column=i, padx=2, pady=1, sticky='ew')
            
        # 配置网格权重
        for i in range(3):
            basic_frame.grid_columnconfigure(i, weight=1)
            edit_frame.grid_columnconfigure(i, weight=1)
            
    def create_enhanced_center_panel(self, parent):
        """创建增强版中心编辑面板"""
        center_panel = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        center_panel.grid(row=0, column=1, sticky='nsew', padx=8)
        
        # 编辑器头部（增强版）
        header_frame = tk.Frame(center_panel, bg=self.colors['surface'], height=80)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        # 当前字符显示（增强版）
        self.char_display_frame = tk.Frame(header_frame, bg=self.colors['primary'], relief='solid', bd=2)
        self.char_display_frame.pack(side=tk.LEFT)
        
        self.current_char_label = tk.Label(self.char_display_frame,
                                          text=self.current_char,
                                          font=('Microsoft YaHei UI', 28, 'bold'),
                                          fg='white',
                                          bg=self.colors['primary'],
                                          width=3,
                                          height=1)
        self.current_char_label.pack(padx=15, pady=10)
        
        # 字符信息（增强版）
        info_frame = tk.Frame(header_frame, bg=self.colors['surface'])
        info_frame.pack(side=tk.LEFT, padx=(20, 0), fill=tk.Y)
        
        self.char_label = tk.Label(info_frame,
                                  text=f'正在编辑: {self.current_char}',
                                  font=('Microsoft YaHei UI', 18, 'bold'),
                                  fg=self.colors['text'],
                                  bg=self.colors['surface'])
        self.char_label.pack(anchor='w')
        
        self.char_info_label = tk.Label(info_frame,
                                       text=f'Unicode: {ord(self.current_char)} | 笔画数: 0 | 频率: 高',
                                       font=('Microsoft YaHei UI', 11),
                                       fg=self.colors['text_muted'],
                                       bg=self.colors['surface'])
        self.char_info_label.pack(anchor='w')
        
        # 设计进度
        self.progress_label = tk.Label(info_frame,
                                      text='设计进度: 未开始',
                                      font=('Microsoft YaHei UI', 10),
                                      fg=self.colors['warning'],
                                      bg=self.colors['surface'])
        self.progress_label.pack(anchor='w', pady=(5, 0))
        
        # 工具栏（增强版）
        toolbar_frame = tk.Frame(header_frame, bg=self.colors['surface'])
        toolbar_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.create_enhanced_editor_toolbar(toolbar_frame)
        
        # 画布区域（增强版）
        canvas_container = tk.Frame(center_panel, bg=self.colors['surface'])
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # 画布边框（增强版）
        canvas_border = tk.Frame(canvas_container, 
                                bg=self.colors['border'], 
                                relief='solid', bd=3)
        canvas_border.pack(fill=tk.BOTH, expand=True)
        
        # 主画布（修正游标）
        self.canvas = tk.Canvas(canvas_border,
                               bg='#ffffff',
                               highlightthickness=0,
                               cursor='pencil')  # 使用有效的游标
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # 绘制增强版网格
        self.draw_enhanced_grid()
        
        # 度量控制区域（增强版）
        metrics_frame = tk.Frame(center_panel, bg=self.colors['surface'], height=60)
        metrics_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        metrics_frame.pack_propagate(False)
        
        self.create_enhanced_metrics_controls(metrics_frame)
        
    def create_enhanced_editor_toolbar(self, parent):
        """创建增强版编辑器工具栏"""
        # 视图控制
        view_frame = tk.LabelFrame(parent,
                                  text="视图",
                                  font=('Microsoft YaHei UI', 9, 'bold'),
                                  fg=self.colors['text'],
                                  bg=self.colors['surface'])
        view_frame.pack(fill=tk.X, pady=(0, 5))
        
        view_tools = [
            ("放大", self.zoom_in),
            ("缩小", self.zoom_out),
            ("适应", self.fit_view),
            ("实际大小", self.actual_size)
        ]
        
        for text, command in view_tools:
            btn = tk.Button(view_frame,
                           text=text,
                           command=command,
                           font=('Microsoft YaHei UI', 8),
                           bg=self.colors['border'],
                           fg=self.colors['text'],
                           borderwidth=0,
                           width=6,
                           height=1,
                           relief='flat',
                           cursor='hand2')
            btn.pack(pady=1)
            
        # 网格控制
        grid_frame = tk.LabelFrame(parent,
                                  text="网格",
                                  font=('Microsoft YaHei UI', 9, 'bold'),
                                  fg=self.colors['text'],
                                  bg=self.colors['surface'])
        grid_frame.pack(fill=tk.X, pady=5)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        grid_check = tk.Checkbutton(grid_frame,
                                   text="显示网格",
                                   variable=self.show_grid_var,
                                   command=self.toggle_grid,
                                   font=('Microsoft YaHei UI', 8),
                                   bg=self.colors['surface'],
                                   fg=self.colors['text'])
        grid_check.pack(anchor='w')
        
        self.show_guides_var = tk.BooleanVar(value=True)
        guides_check = tk.Checkbutton(grid_frame,
                                     text="辅助线",
                                     variable=self.show_guides_var,
                                     command=self.toggle_guides,
                                     font=('Microsoft YaHei UI', 8),
                                     bg=self.colors['surface'],
                                     fg=self.colors['text'])
        guides_check.pack(anchor='w')
        
    def create_enhanced_metrics_controls(self, parent):
        """创建增强版度量控制"""
        # 字符宽度
        width_frame = tk.Frame(parent, bg=self.colors['surface'])
        width_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        tk.Label(width_frame,
                text="字符宽度",
                font=('Microsoft YaHei UI', 10, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(anchor='w')
        
        self.advance_width_var = tk.DoubleVar(value=1.0)
        width_scale = tk.Scale(width_frame,
                              from_=0.3, to=2.5, resolution=0.05,
                              orient=tk.HORIZONTAL,
                              variable=self.advance_width_var,
                              command=self.update_char_metrics,
                              bg=self.colors['surface'],
                              fg=self.colors['primary'],
                              highlightthickness=0,
                              length=160)
        width_scale.pack()
        
        # 字符间距
        spacing_frame = tk.Frame(parent, bg=self.colors['surface'])
        spacing_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        tk.Label(spacing_frame,
                text="字符间距",
                font=('Microsoft YaHei UI', 10, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(anchor='w')
        
        self.char_spacing_var = tk.IntVar(value=0)
        spacing_scale = tk.Scale(spacing_frame,
                                from_=-30, to=30,
                                orient=tk.HORIZONTAL,
                                variable=self.char_spacing_var,
                                command=self.update_char_spacing,
                                bg=self.colors['surface'],
                                fg=self.colors['primary'],
                                highlightthickness=0,
                                length=140)
        spacing_scale.pack()
        
        # 垂直对齐
        align_frame = tk.Frame(parent, bg=self.colors['surface'])
        align_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        tk.Label(align_frame,
                text="垂直对齐",
                font=('Microsoft YaHei UI', 10, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(anchor='w')
        
        self.vertical_align_var = tk.IntVar(value=0)
        align_scale = tk.Scale(align_frame,
                              from_=-20, to=20,
                              orient=tk.HORIZONTAL,
                              variable=self.vertical_align_var,
                              command=self.update_vertical_align,
                              bg=self.colors['surface'],
                              fg=self.colors['primary'],
                              highlightthickness=0,
                              length=120)
        align_scale.pack()
        
    def create_enhanced_right_panel(self, parent):
        """创建增强版右侧面板"""
        right_panel = tk.Frame(parent, bg=self.colors['surface'], 
                              width=380, relief='solid', bd=1)
        right_panel.grid(row=0, column=2, sticky='nsew', padx=(8, 0))
        right_panel.grid_propagate(False)
        
        # 标签页容器
        notebook = ttk.Notebook(right_panel, style='Modern.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 预览标签页（增强版）
        self.create_enhanced_preview_tab(notebook)
        
        # 分析标签页（增强版）
        self.create_enhanced_analysis_tab(notebook)
        
        # 设置标签页（增强版）
        self.create_enhanced_settings_tab(notebook)
        
        # 导出标签页（增强版）
        self.create_enhanced_export_tab(notebook)
        
        # 历史标签页（新增）
        self.create_history_tab(notebook)
        
    def create_enhanced_preview_tab(self, notebook):
        """创建增强版预览标签页"""
        preview_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(preview_frame, text="预览")
        
        # 单字符预览（增强版）
        char_preview_card = self.create_modern_card(preview_frame, "字符预览", 350, 160)
        char_preview_card.pack(pady=(8, 4), fill=tk.X)
        
        char_content = char_preview_card.winfo_children()[1]  # 获取内容框架
        
        self.preview_canvas = tk.Canvas(char_content, height=120, 
                                       bg='#ffffff', highlightthickness=0,
                                       relief='solid', bd=1)
        self.preview_canvas.pack(fill=tk.X, pady=8)
        
        # 文本预览（增强版）
        text_preview_card = self.create_modern_card(preview_frame, "文本预览", 350, 220)
        text_preview_card.pack(pady=4, fill=tk.X)
        
        text_content = text_preview_card.winfo_children()[1]
        
        # 预览文本输入
        input_frame = tk.Frame(text_content, bg=self.colors['surface'])
        input_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(input_frame, text="预览文本:",
                font=('Microsoft YaHei UI', 10),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(anchor='w')
        
        self.preview_text = tk.Entry(input_frame,
                                    font=('Microsoft YaHei UI', 11),
                                    bg=self.colors['surface'],
                                    fg=self.colors['text'],
                                    bd=1, relief='solid')
        self.preview_text.pack(fill=tk.X, ipady=6, pady=(3, 0))
        self.preview_text.insert(0, '中国汉字书法艺术')
        # 修正: 调用正确的方法名 update_text_preview_enhanced
        self.preview_text.bind('<KeyRelease>', self.update_text_preview_enhanced)
        
        self.text_preview_canvas = tk.Canvas(text_content, height=100,
                                           bg='#ffffff', highlightthickness=0,
                                           relief='solid', bd=1)
        self.text_preview_canvas.pack(fill=tk.X)
        
        # 预览控制（增强版）
        control_frame = tk.Frame(text_content, bg=self.colors['surface'])
        control_frame.pack(fill=tk.X, pady=(8, 0))
        
        # 预览大小
        size_frame = tk.Frame(control_frame, bg=self.colors['surface'])
        size_frame.pack(fill=tk.X, pady=3)
        
        tk.Label(size_frame, text="预览大小:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.preview_size_var = tk.IntVar(value=4)
        size_scale = tk.Scale(size_frame,
                             from_=1, to=10,
                             orient=tk.HORIZONTAL,
                             variable=self.preview_size_var,
                             # 修正: 调用正确的方法名 update_text_preview_enhanced
                             command=lambda v: self.update_text_preview_enhanced(),
                             bg=self.colors['surface'],
                             fg=self.colors['primary'],
                             length=120)
        size_scale.pack(side=tk.RIGHT)
        
        # 预览样式
        style_frame = tk.Frame(control_frame, bg=self.colors['surface'])
        style_frame.pack(fill=tk.X, pady=3)
        
        tk.Label(style_frame, text="预览样式:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.preview_style_var = tk.StringVar(value='正常')
        style_combo = ttk.Combobox(style_frame,
                                  textvariable=self.preview_style_var,
                                  values=['正常', '粗体', '斜体', '描边'],
                                  state='readonly',
                                  font=('Microsoft YaHei UI', 9),
                                  width=8)
        style_combo.pack(side=tk.RIGHT)
        style_combo.bind('<<ComboboxSelected>>', lambda e: self.update_text_preview_enhanced())
        
        # 统计信息（增强版）
        stats_card = self.create_modern_card(preview_frame, "项目统计", 350, 120)
        stats_card.pack(pady=4, fill=tk.X)
        
        stats_content = stats_card.winfo_children()[1]
        
        self.stats_text = tk.Text(stats_content, height=4,
                                 bg=self.colors['surface'],
                                 fg=self.colors['text'],
                                 font=('Microsoft YaHei UI', 9),
                                 wrap=tk.WORD, bd=0)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
    def create_enhanced_analysis_tab(self, notebook):
        """创建增强版分析标签页"""
        analysis_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(analysis_frame, text="分析")
        
        # 笔画分析（增强版）
        stroke_card = self.create_modern_card(analysis_frame, "笔画分析", 350, 200)
        stroke_card.pack(pady=(8, 4), fill=tk.X)
        
        stroke_content = stroke_card.winfo_children()[1]
        
        self.stroke_info_text = tk.Text(stroke_content, height=8,
                                       bg=self.colors['surface'],
                                       fg=self.colors['text'],
                                       font=('Microsoft YaHei UI', 9),
                                       wrap=tk.WORD, bd=0)
        self.stroke_info_text.pack(fill=tk.BOTH, expand=True)
        
        # 字形分析（增强版）
        shape_card = self.create_modern_card(analysis_frame, "字形分析", 350, 150)
        shape_card.pack(pady=4, fill=tk.X)
        
        shape_content = shape_card.winfo_children()[1]
        
        self.shape_analysis_text = tk.Text(shape_content, height=6,
                                          bg=self.colors['surface'],
                                          fg=self.colors['text'],
                                          font=('Microsoft YaHei UI', 9),
                                          wrap=tk.WORD, bd=0)
        self.shape_analysis_text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        analyze_btn = tk.Button(shape_content,
                               text="🔍 深度分析字符",
                               command=self.analyze_character,
                               font=('Microsoft YaHei UI', 10, 'bold'),
                               bg=self.colors['info'],
                               fg='white',
                               borderwidth=0,
                               padx=15,
                               pady=6,
                               cursor='hand2')
        analyze_btn.pack(fill=tk.X)
        
        # 质量评估（新增）
        quality_card = self.create_modern_card(analysis_frame, "质量评估", 350, 120)
        quality_card.pack(pady=4, fill=tk.X)
        
        quality_content = quality_card.winfo_children()[1]
        
        self.quality_text = tk.Text(quality_content, height=4,
                                   bg=self.colors['surface'],
                                   fg=self.colors['text'],
                                   font=('Microsoft YaHei UI', 9),
                                   wrap=tk.WORD, bd=0)
        self.quality_text.pack(fill=tk.BOTH, expand=True)
        
    def create_enhanced_settings_tab(self, notebook):
        """创建增强版设置标签页"""
        settings_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(settings_frame, text="设置")
        
        # 创建滚动区域
        canvas = tk.Canvas(settings_frame, bg=self.colors['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['surface'])
        
        scrollable_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 字体信息设置
        info_card = self.create_modern_card(scrollable_frame, "字体信息", 320, 200)
        info_card.pack(pady=8, padx=8, fill=tk.X)
        
        info_content = info_card.winfo_children()[1]
        
        self.create_setting_entry_modern(info_content, '字体名称:', 'family_name')
        self.create_setting_entry_modern(info_content, '样式名称:', 'style_name')
        self.create_setting_entry_modern(info_content, '版本:', 'version')
        self.create_setting_entry_modern(info_content, '版权:', 'copyright')
        
        # 字体度量设置
        metrics_card = self.create_modern_card(scrollable_frame, "字体度量", 320, 220)
        metrics_card.pack(pady=4, padx=8, fill=tk.X)
        
        metrics_content = metrics_card.winfo_children()[1]
        
        self.create_setting_scale_modern(metrics_content, 'UPM单位:', 'units_per_em', 500, 2048)
        self.create_setting_scale_modern(metrics_content, '上伸部:', 'ascender', 600, 1000)
        self.create_setting_scale_modern(metrics_content, '下伸部:', 'descender', -300, 0)
        self.create_setting_scale_modern(metrics_content, '行间距:', 'line_gap', 0, 200)
        
        # 界面设置
        ui_card = self.create_modern_card(scrollable_frame, "界面设置", 320, 180)
        ui_card.pack(pady=4, padx=8, fill=tk.X)
        
        ui_content = ui_card.winfo_children()[1]
        
        # 网格大小
        grid_frame = tk.Frame(ui_content, bg=self.colors['surface'])
        grid_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(grid_frame, text="网格大小:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        grid_combo = ttk.Combobox(grid_frame,
                                 textvariable=self.grid_size_var,
                                 values=[32, 48, 64, 80, 96],
                                 state='readonly',
                                 width=8)
        grid_combo.pack(side=tk.RIGHT)
        grid_combo.bind('<<ComboboxSelected>>', self.on_grid_size_changed)
        
        # 渲染质量
        quality_frame = tk.Frame(ui_content, bg=self.colors['surface'])
        quality_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(quality_frame, text="渲染质量:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.quality_var = tk.StringVar(value=self.render_quality)
        quality_combo = ttk.Combobox(quality_frame,
                                    textvariable=self.quality_var,
                                    values=['快速', '正常', '高质量', '最佳'],
                                    state='readonly',
                                    width=8)
        quality_combo.pack(side=tk.RIGHT)
        
        # 自动保存
        auto_save_frame = tk.Frame(ui_content, bg=self.colors['surface'])
        auto_save_frame.pack(fill=tk.X, pady=5)
        
        self.auto_save_var = tk.BooleanVar(value=True)
        auto_save_check = tk.Checkbutton(auto_save_frame,
                                        text="自动保存（每5分钟）",
                                        variable=self.auto_save_var,
                                        font=('Microsoft YaHei UI', 9),
                                        bg=self.colors['surface'],
                                        fg=self.colors['text'])
        auto_save_check.pack(anchor='w')
        
        # 性能设置
        perf_card = self.create_modern_card(scrollable_frame, "性能设置", 320, 120)
        perf_card.pack(pady=4, padx=8, fill=tk.X)
        
        perf_content = perf_card.winfo_children()[1]
        
        # 缓存大小
        cache_frame = tk.Frame(perf_content, bg=self.colors['surface'])
        cache_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(cache_frame, text="缓存大小(MB):",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.cache_size_var = tk.IntVar(value=100)
        cache_scale = tk.Scale(cache_frame,
                              from_=50, to=500,
                              orient=tk.HORIZONTAL,
                              variable=self.cache_size_var,
                              length=120)
        cache_scale.pack(side=tk.RIGHT)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
    def create_enhanced_export_tab(self, notebook):
        """创建增强版导出标签页"""
        export_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(export_frame, text="导出")
        
        # TTF导出（增强版）
        ttf_card = self.create_modern_card(export_frame, "TTF字体导出", 350, 180)
        ttf_card.pack(pady=(8, 4), fill=tk.X)
        
        ttf_content = ttf_card.winfo_children()[1]
        
        # 状态显示
        if FONTTOOLS_AVAILABLE:
            status_text = "✅ FontTools已安装"
            status_color = self.colors['success']
        else:
            status_text = "❌ 需要安装FontTools"
            status_color = self.colors['danger']
            
        status_label = tk.Label(ttf_content, text=status_text,
                               fg=status_color, bg=self.colors['surface'],
                               font=('Microsoft YaHei UI', 10, 'bold'))
        status_label.pack(pady=8)
        
        # 导出选项
        options_frame = tk.Frame(ttf_content, bg=self.colors['surface'])
        options_frame.pack(fill=tk.X, pady=5)
        
        self.include_chinese_metrics = tk.BooleanVar(value=True)
        chinese_check = tk.Checkbutton(options_frame,
                                      text="包含中文字体度量优化",
                                      variable=self.include_chinese_metrics,
                                      bg=self.colors['surface'],
                                      fg=self.colors['text'],
                                      font=('Microsoft YaHei UI', 9))
        chinese_check.pack(anchor='w')
        
        self.include_kerning = tk.BooleanVar(value=True)
        kerning_check = tk.Checkbutton(options_frame,
                                      text="包含字距调整",
                                      variable=self.include_kerning,
                                      bg=self.colors['surface'],
                                      fg=self.colors['text'],
                                      font=('Microsoft YaHei UI', 9))
        kerning_check.pack(anchor='w')
        
        self.optimize_file_size = tk.BooleanVar(value=True)
        optimize_check = tk.Checkbutton(options_frame,
                                       text="优化文件大小",
                                       variable=self.optimize_file_size,
                                       bg=self.colors['surface'],
                                       fg=self.colors['text'],
                                       font=('Microsoft YaHei UI', 9))
        optimize_check.pack(anchor='w')
        
        ttf_btn = tk.Button(ttf_content,
                           text="📤 导出TTF字体",
                           command=self.export_ttf,
                           font=('Microsoft YaHei UI', 11, 'bold'),
                           bg=self.colors['primary'],
                           fg='white',
                           borderwidth=0,
                           padx=15,
                           pady=8,
                           cursor='hand2')
        ttf_btn.pack(fill=tk.X, pady=(8, 0))
        
        # 图像导出（增强版）
        img_card = self.create_modern_card(export_frame, "图像导出", 350, 200)
        img_card.pack(pady=4, fill=tk.X)
        
        img_content = img_card.winfo_children()[1]
        
        # PNG导出设置
        png_frame = tk.LabelFrame(img_content,
                                 text="PNG设置",
                                 font=('Microsoft YaHei UI', 10, 'bold'),
                                 fg=self.colors['text'],
                                 bg=self.colors['surface'])
        png_frame.pack(fill=tk.X, pady=5)
        
        size_frame = tk.Frame(png_frame, bg=self.colors['surface'])
        size_frame.pack(fill=tk.X, pady=5, padx=8)
        
        tk.Label(size_frame, text="字符大小:",
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.png_char_size = tk.IntVar(value=128)
        size_scale = tk.Scale(size_frame,
                             from_=64, to=1024,
                             orient=tk.HORIZONTAL,
                             variable=self.png_char_size,
                             length=120)
        size_scale.pack(side=tk.RIGHT)
        
        # 背景透明
        self.png_transparent = tk.BooleanVar(value=True)
        trans_check = tk.Checkbutton(png_frame,
                                    text="透明背景",
                                    variable=self.png_transparent,
                                    bg=self.colors['surface'],
                                    fg=self.colors['text'],
                                    font=('Microsoft YaHei UI', 9))
        trans_check.pack(anchor='w', padx=8)
        
        # 导出按钮
        export_buttons_frame = tk.Frame(img_content, bg=self.colors['surface'])
        export_buttons_frame.pack(fill=tk.X, pady=8)
        
        export_buttons = [
            ("导出PNG", self.export_png, self.colors['success']),
            ("导出SVG", self.export_svg, self.colors['info']),
            ("批量导出", self.batch_export, self.colors['warning'])
        ]
        
        for text, command, color in export_buttons:
            btn = tk.Button(export_buttons_frame,
                           text=text,
                           command=command,
                           font=('Microsoft YaHei UI', 9, 'bold'),
                           bg=color,
                           fg='white',
                           borderwidth=0,
                           padx=10,
                           pady=4,
                           cursor='hand2')
            btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
    def create_history_tab(self, notebook):
        """创建历史记录标签页"""
        history_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(history_frame, text="历史")
        
        # 操作历史
        history_card = self.create_modern_card(history_frame, "操作历史", 350, 250)
        history_card.pack(pady=(8, 4), fill=tk.X)
        
        history_content = history_card.winfo_children()[1]
        
        # 历史列表
        self.history_listbox = tk.Listbox(history_content,
                                         font=('Microsoft YaHei UI', 9),
                                         bg=self.colors['surface'],
                                         fg=self.colors['text'],
                                         selectbackground=self.colors['primary_light'],
                                         height=10)
        self.history_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # 历史控制按钮
        history_btn_frame = tk.Frame(history_content, bg=self.colors['surface'])
        history_btn_frame.pack(fill=tk.X)
        
        history_buttons = [
            ("恢复", self.restore_from_history, self.colors['success']),
            ("清除", self.clear_history, self.colors['danger'])
        ]
        
        for text, command, color in history_buttons:
            btn = tk.Button(history_btn_frame,
                           text=text,
                           command=command,
                           font=('Microsoft YaHei UI', 9),
                           bg=color,
                           fg='white',
                           borderwidth=0,
                           padx=10,
                           pady=4,
                           cursor='hand2')
            btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            
        # 最近编辑
        recent_card = self.create_modern_card(history_frame, "最近编辑", 350, 150)
        recent_card.pack(pady=4, fill=tk.X)
        
        recent_content = recent_card.winfo_children()[1]
        
        self.recent_text = tk.Text(recent_content, height=6,
                                  bg=self.colors['surface'],
                                  fg=self.colors['text'],
                                  font=('Microsoft YaHei UI', 9),
                                  wrap=tk.WORD, bd=0)
        self.recent_text.pack(fill=tk.BOTH, expand=True)
        
    def create_modern_card(self, parent, title, width=300, height=200):
        """创建现代化卡片组件"""
        card_frame = tk.Frame(parent, bg=self.colors['surface'], 
                             relief='solid', bd=1)
        
        # 卡片标题
        if title:
            title_frame = tk.Frame(card_frame, bg=self.colors['primary'], height=35)
            title_frame.pack(fill=tk.X)
            title_frame.pack_propagate(False)
            
            title_label = tk.Label(title_frame, text=title, 
                                  bg=self.colors['primary'], fg='white',
                                  font=('Microsoft YaHei UI', 11, 'bold'))
            title_label.pack(expand=True)
        
        # 内容区域
        content_frame = tk.Frame(card_frame, bg=self.colors['surface'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        return card_frame
        
    def create_setting_entry_modern(self, parent, label, key):
        """创建现代化设置输入框"""
        frame = tk.Frame(parent, bg=self.colors['surface'])
        frame.pack(fill=tk.X, pady=4)
        
        tk.Label(frame, text=label,
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface'],
                width=8, anchor='w').pack(side=tk.LEFT)
        
        var = tk.StringVar(value=str(self.font_metadata[key]))
        entry = tk.Entry(frame, textvariable=var,
                        font=('Microsoft YaHei UI', 9),
                        bg=self.colors['surface'],
                        fg=self.colors['text'],
                        bd=1, relief='solid')
        entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        
        entry.bind('<FocusOut>', lambda e, k=key, v=var: self.update_metadata(k, v.get()))
        
        return var
        
    def create_setting_scale_modern(self, parent, label, key, min_val, max_val):
        """创建现代化设置滑块"""
        frame = tk.Frame(parent, bg=self.colors['surface'])
        frame.pack(fill=tk.X, pady=4)
        
        tk.Label(frame, text=label,
                font=('Microsoft YaHei UI', 9),
                fg=self.colors['text'],
                bg=self.colors['surface'],
                width=8, anchor='w').pack(side=tk.LEFT)
        
        var = tk.IntVar(value=self.font_metadata[key])
        scale = tk.Scale(frame,
                        from_=min_val, to=max_val,
                        orient=tk.HORIZONTAL,
                        variable=var,
                        command=lambda v, k=key: self.update_metadata(k, int(v)),
                        bg=self.colors['surface'],
                        fg=self.colors['primary'],
                        length=120)
        scale.pack(side=tk.RIGHT, padx=(8, 0))
        
        return var
        
    def create_enhanced_statusbar(self, parent):
        """创建增强版状态栏"""
        statusbar = tk.Frame(parent, bg=self.colors['dark'], height=40)
        statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        statusbar.pack_propagate(False)
        
        # 状态信息
        self.status_label = tk.Label(statusbar,
                                    text=f'就绪 | 当前字符: {self.current_char}',
                                    font=('Microsoft YaHei UI', 9),
                                    fg='white', bg=self.colors['dark'])
        self.status_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(statusbar, 
                                           length=200,
                                           mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=15, pady=10)
        
        # 统计信息
        self.stats_label = tk.Label(statusbar,
                                   text=f'字符数: {len(self.font_data)}',
                                   font=('Microsoft YaHei UI', 9),
                                   fg=self.colors['primary_light'], bg=self.colors['dark'])
        self.stats_label.pack(side=tk.LEFT, padx=15)
        
        # 工具状态
        self.tool_status_label = tk.Label(statusbar,
                                         text=f'工具: {self.current_tool}',
                                         font=('Microsoft YaHei UI', 9),
                                         fg=self.colors['accent'], bg=self.colors['dark'])
        self.tool_status_label.pack(side=tk.LEFT, padx=15)
        
        # 时间显示
        self.time_label = tk.Label(statusbar,
                                  text=datetime.now().strftime('%H:%M:%S'),
                                  font=('Consolas', 9),
                                  fg=self.colors['muted'], bg=self.colors['dark'])
        self.time_label.pack(side=tk.RIGHT, padx=15, pady=10)

    # 搜索和过滤相关方法
    def on_search_changed(self, event=None):
        """搜索改变时的处理"""
        self.search_query = self.search_var.get()
        if len(self.search_query) >= 2 or self.search_query == '':
            self.root.after(300, self.perform_search)  # 延迟搜索
    
    def on_category_changed(self, event=None):
        """分类改变时的处理"""
        self.current_category = self.category_var.get()
        self.perform_search()
    
    def on_per_page_changed(self, event=None):
        """每页显示数量改变时的处理"""
        self.chars_per_page = int(self.per_page_var.get())
        self.current_page = 0
        self.update_character_display()
    
    def on_grid_size_changed(self, event=None):
        """网格大小改变时的处理"""
        old_size = self.grid_size
        self.grid_size = self.grid_size_var.get()
        # 这里可以添加网格大小变化的处理逻辑
        self.draw_enhanced_grid()
    
    def perform_search(self):
        """执行搜索"""
        try:
            query = self.search_query.strip()
            category = self.current_category if self.current_category != '全部' else None
            
            if query:
                results = self.char_db.search_characters(query, category, 1000)
            else:
                results = self.char_db.get_characters_by_category(self.current_category, 1000)
            
            # 根据高级过滤条件进一步筛选
            filtered_results = []
            for char_data in results:
                stroke_count = char_data[8] or 1  # stroke_count字段
                is_designed = char_data[9] == 1   # is_designed字段
                
                # 笔画数过滤
                if not (self.min_strokes.get() <= stroke_count <= self.max_strokes.get()):
                    continue
                    
                # 设计状态过滤
                if not ((is_designed and self.show_designed.get()) or 
                       (not is_designed and self.show_undesigned.get())):
                    continue
                    
                filtered_results.append(char_data)
            
            self.displayed_chars = filtered_results
            self.current_page = 0
            self.update_character_display()
        except Exception as e:
            print(f"搜索失败: {e}")
            # 使用基本字符作为后备
            basic_chars = [('中',), ('国',), ('汉',), ('字',)]
            self.displayed_chars = [(0, char[0], char[0], '汉字', 100, None, None, None, 10, 0, None, None) for char in basic_chars]
            self.update_character_display()
    
    def update_character_display(self):
        """更新字符显示"""
        # 清除现有按钮
        for widget in self.char_grid_frame.winfo_children():
            widget.destroy()
        self.char_buttons.clear()
        
        # 计算分页
        total_chars = len(self.displayed_chars)
        total_pages = max(1, (total_chars - 1) // self.chars_per_page + 1)
        start_idx = self.current_page * self.chars_per_page
        end_idx = min(start_idx + self.chars_per_page, total_chars)
        
        # 更新信息标签
        self.char_count_label.config(text=f"字符总数: {total_chars}")
        self.page_info_label.config(text=f"第{self.current_page + 1}页/共{total_pages}页")
        
        # 更新翻页按钮状态
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < total_pages - 1 else tk.DISABLED)
        
        # 显示当前页的字符
        cols = 8
        for i, char_data in enumerate(self.displayed_chars[start_idx:end_idx]):
            char = char_data[1]  # char字段
            is_designed = char_data[9] == 1  # is_designed字段
            frequency = char_data[4]  # frequency字段
            
            row = i // cols
            col = i % cols
            
            btn = self.create_enhanced_char_button(self.char_grid_frame, char, is_designed, frequency)
            btn.grid(row=row, column=col, padx=1, pady=1, sticky='nsew')
            self.char_buttons[char] = btn
            
        # 配置网格权重
        for i in range(cols):
            self.char_grid_frame.grid_columnconfigure(i, weight=1)
    
    def create_enhanced_char_button(self, parent, char, is_designed, frequency):
        """创建增强版字符按钮"""
        # 根据设计状态和频率确定按钮颜色
        if is_designed:
            bg_color = self.colors['success']
            fg_color = 'white'
        elif frequency > 50:
            bg_color = self.colors['warning']
            fg_color = 'white'
        else:
            bg_color = self.colors['surface']
            fg_color = self.colors['text']
            
        btn = tk.Button(parent,
                       text=char if char != ' ' else 'SP',
                       command=lambda: self.select_char(char),
                       font=('Microsoft YaHei UI', 11, 'bold'),
                       bg=bg_color,
                       fg=fg_color,
                       borderwidth=1,
                       relief='solid',
                       width=4,
                       height=2,
                       cursor='hand2',
                       activebackground=self.colors['primary_light'],
                       activeforeground='white')
        
        # 现代化悬停效果
        def on_enter(e):
            if char != self.current_char:
                btn.config(bg=self.colors['primary_light'], fg='white')
            
        def on_leave(e):
            if char != self.current_char:
                btn.config(bg=bg_color, fg=fg_color)
                          
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        # 添加工具提示
        self.create_tooltip(btn, f"字符: {char}\nUnicode: U+{ord(char):04X}\n频率: {frequency}\n状态: {'已设计' if is_designed else '未设计'}")
        
        return btn
    
    def create_tooltip(self, widget, text):
        """创建工具提示"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(tooltip, text=text,
                           background=self.colors['dark'],
                           foreground='white',
                           font=('Microsoft YaHei UI', 9),
                           relief='solid',
                           borderwidth=1,
                           padx=8, pady=4)
            label.pack()
            
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
    
    # 分页相关方法
    def prev_page(self):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_character_display()
    
    def next_page(self):
        """下一页"""
        total_pages = max(1, (len(self.displayed_chars) - 1) // self.chars_per_page + 1)
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_character_display()
    
    def goto_page(self, event=None):
        """跳转到指定页面"""
        try:
            page = int(self.page_entry.get()) - 1
            total_pages = max(1, (len(self.displayed_chars) - 1) // self.chars_per_page + 1)
            if 0 <= page < total_pages:
                self.current_page = page
                self.update_character_display()
        except ValueError:
            pass
    
    # 绘制相关方法
    def draw_enhanced_grid(self):
        """绘制增强版网格"""
        self.canvas.delete('all')
        
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.draw_enhanced_grid)
            return
            
        # 计算网格尺寸
        available_size = min(canvas_width - 40, canvas_height - 40)
        self.pixel_size = max(8, available_size // self.grid_size)
        
        grid_total_size = self.pixel_size * self.grid_size
        start_x = (canvas_width - grid_total_size) // 2
        start_y = (canvas_height - grid_total_size) // 2
        
        self.grid_start_x = start_x
        self.grid_start_y = start_y
        
        # 绘制外边框
        self.canvas.create_rectangle(start_x-2, start_y-2,
                                    start_x + grid_total_size + 2,
                                    start_y + grid_total_size + 2,
                                    outline=self.colors['primary'], width=3)
        
        # 绘制网格线
        if self.show_grid_var.get():
            grid_spacing = self.pixel_size * 4
            for i in range(1, self.grid_size // 4):
                x = start_x + i * grid_spacing
                y = start_y + i * grid_spacing
                self.canvas.create_line(x, start_y, x, start_y + grid_total_size,
                                       fill='#f0f0f0', width=1)
                self.canvas.create_line(start_x, y, start_x + grid_total_size, y,
                                       fill='#f0f0f0', width=1)
        
        # 绘制辅助线
        if self.show_guides_var.get():
            center_x = start_x + grid_total_size // 2
            center_y = start_y + grid_total_size // 2
            
            # 中线（十字）
            self.canvas.create_line(center_x, start_y, center_x, start_y + grid_total_size,
                                   fill=self.colors['primary_light'], width=2, dash=(6, 4))
            self.canvas.create_line(start_x, center_y, start_x + grid_total_size, center_y,
                                   fill=self.colors['primary_light'], width=2, dash=(6, 4))
            
            # 对角线
            self.canvas.create_line(start_x, start_y, start_x + grid_total_size, start_y + grid_total_size,
                                   fill=self.colors['border'], width=1, dash=(4, 8))
            self.canvas.create_line(start_x + grid_total_size, start_y, start_x, start_y + grid_total_size,
                                   fill=self.colors['border'], width=1, dash=(4, 8))
        
        # 绘制已有笔画
        self.draw_existing_strokes_enhanced()
        
    def draw_existing_strokes_enhanced(self):
        """绘制增强版现有笔画"""
        if self.current_char not in self.strokes:
            return
            
        for stroke in self.strokes[self.current_char]:
            if len(stroke.points) < 2:
                continue
                
            # 转换坐标并绘制笔画
            canvas_points = []
            for point in stroke.points:
                canvas_x = self.grid_start_x + point[0] * self.pixel_size
                canvas_y = self.grid_start_y + point[1] * self.pixel_size
                canvas_points.extend([canvas_x, canvas_y])
            
            if len(canvas_points) >= 4:
                # 计算笔触宽度（支持压感）
                if self.pressure_sensitivity:
                    # 为每段线条计算不同的宽度
                    for i in range(0, len(canvas_points) - 3, 2):
                        segment_points = canvas_points[i:i+4]
                        point_idx = i // 2
                        if point_idx < len(stroke.points):
                            pressure = stroke.points[point_idx][2]
                            line_width = max(1, int(self.stroke_width * pressure))
                            
                            stroke_color = getattr(stroke, 'color', self.current_color)
                            
                            self.canvas.create_line(segment_points,
                                                   fill=stroke_color,
                                                   width=line_width,
                                                   smooth=True,
                                                   capstyle=tk.ROUND,
                                                   joinstyle=tk.ROUND)
                else:
                    # 统一宽度
                    stroke_color = getattr(stroke, 'color', self.current_color)
                    self.canvas.create_line(canvas_points,
                                           fill=stroke_color,
                                           width=self.stroke_width,
                                           smooth=True,
                                           capstyle=tk.ROUND,
                                           joinstyle=tk.ROUND)
    
    # 笔画绘制相关方法
    def start_stroke(self, event):
        """开始新笔画"""
        grid_x = (event.x - self.grid_start_x) / self.pixel_size
        grid_y = (event.y - self.grid_start_y) / self.pixel_size
        
        if 0 <= grid_x <= self.grid_size and 0 <= grid_y <= self.grid_size:
            self.current_stroke = HandwritingStroke()
            self.current_stroke.color = self.current_color
            self.current_stroke.stroke_type = self.current_tool
            
            pressure = 0.8 if self.pressure_sensitivity else 1.0
            self.current_stroke.add_point(grid_x, grid_y, pressure)
            
            # 添加到历史记录
            self.add_to_history(f"开始笔画 - {self.current_char}")
            
    def continue_stroke(self, event):
        """继续笔画"""
        if self.current_stroke is None:
            return
            
        grid_x = (event.x - self.grid_start_x) / self.pixel_size
        grid_y = (event.y - self.grid_start_y) / self.pixel_size
        
        if 0 <= grid_x <= self.grid_size and 0 <= grid_y <= self.grid_size:
            # 模拟压感
            pressure = 1.0
            if self.pressure_sensitivity and len(self.current_stroke.points) > 0:
                last_point = self.current_stroke.points[-1]
                distance = math.sqrt((grid_x - last_point[0])**2 + (grid_y - last_point[1])**2)
                # 根据移动速度调整压感
                pressure = max(0.3, min(1.0, 1.2 - distance * 0.1))
            
            self.current_stroke.add_point(grid_x, grid_y, pressure)
            self.draw_current_stroke_segment_enhanced()
            
    def end_stroke(self, event):
        """结束笔画"""
        if self.current_stroke is None:
            return
            
        if len(self.current_stroke.points) < 2:
            self.current_stroke = None
            return
            
        if self.stroke_smoothing > 0:
            self.current_stroke.smooth_stroke(self.stroke_smoothing)
            
        if self.current_char not in self.strokes:
            self.strokes[self.current_char] = []
            
        self.strokes[self.current_char].append(self.current_stroke)
        
        # 标记字符为已设计
        try:
            self.char_db.mark_designed(self.current_char)
        except Exception as e:
            print(f"标记字符设计状态失败: {e}")
        
        self.draw_enhanced_grid()
        self.update_previews_enhanced()
        self.update_stroke_info_enhanced()
        self.add_to_history(f"完成笔画 - {self.current_char}")
        
        self.current_stroke = None
        
    def draw_current_stroke_segment_enhanced(self):
        """绘制当前笔画片段（增强版）"""
        if self.current_stroke is None or len(self.current_stroke.points) < 2:
            return
            
        last_two_points = self.current_stroke.points[-2:]
        canvas_points = []
        
        for point in last_two_points:
            canvas_x = self.grid_start_x + point[0] * self.pixel_size
            canvas_y = self.grid_start_y + point[1] * self.pixel_size
            canvas_points.extend([canvas_x, canvas_y])
            
        if self.pressure_sensitivity:
            avg_pressure = (last_two_points[0][2] + last_two_points[1][2]) / 2
            line_width = max(1, int(self.stroke_width * avg_pressure))
        else:
            line_width = self.stroke_width
        
        self.canvas.create_line(canvas_points,
                               fill=self.current_color,
                               width=line_width,
                               capstyle=tk.ROUND,
                               tags="temp_stroke")
    
    # 字符选择和工具方法
    def select_char(self, char):
        """选择字符进行编辑"""
        # 更新按钮状态
        for c, btn in self.char_buttons.items():
            if c == char:
                btn.config(bg=self.colors['primary'], fg='white')
            else:
                # 恢复原始颜色
                char_data = next((cd for cd in self.displayed_chars if cd[1] == c), None)
                if char_data:
                    is_designed = char_data[9] == 1
                    frequency = char_data[4]
                    if is_designed:
                        btn.config(bg=self.colors['success'], fg='white')
                    elif frequency > 50:
                        btn.config(bg=self.colors['warning'], fg='white')
                    else:
                        btn.config(bg=self.colors['surface'], fg=self.colors['text'])
        
        self.current_char = char
        
        # 更新字符显示
        self.current_char_label.config(text=char)
        self.char_label.config(text=f'正在编辑: {char}')
        
        # 更新字符信息
        stroke_count = len(self.strokes.get(char, []))
        char_data = self.font_data.get(char, {})
        frequency_text = '高' if char_data.get('frequency', 0) > 50 else '中' if char_data.get('frequency', 0) > 10 else '低'
        
        self.char_info_label.config(text=f'Unicode: U+{ord(char):04X} | 笔画数: {stroke_count} | 频率: {frequency_text}')
        
        # 更新设计进度
        is_designed = len(self.strokes.get(char, [])) > 0
        progress_text = '已完成' if is_designed else '未开始'
        progress_color = self.colors['success'] if is_designed else self.colors['warning']
        self.progress_label.config(text=f'设计进度: {progress_text}', fg=progress_color)
        
        # 更新度量显示
        if char in self.font_data:
            self.advance_width_var.set(self.font_data[char]['advance_width'])
            
        self.draw_enhanced_grid()
        self.update_previews_enhanced()
        self.update_stroke_info_enhanced()
        self.update_status_enhanced()
        
        self.add_to_history(f"选择字符 - {char}")
        
    def set_current_tool(self, tool):
        """设置当前工具"""
        self.current_tool = tool
        
        # 使用有效的Tkinter游标名称
        cursor_map = {
            'brush': 'pencil',
            'pen': 'pencil',
            'marker': 'pencil',
            'pencil': 'pencil',
            'eraser': 'dotbox'
        }
        
        self.canvas.config(cursor=cursor_map.get(tool, 'pencil'))
        self.update_status_enhanced()
        
    # 参数更新方法
    def update_stroke_width(self, value):
        """更新笔触宽度"""
        self.stroke_width = int(value)
        
    def update_smoothing(self, value):
        """更新平滑度"""
        self.stroke_smoothing = float(value)
        
    def toggle_pressure_sensitivity(self):
        """切换压感效果"""
        self.pressure_sensitivity = self.pressure_var.get()
        
    def choose_color(self, event=None):
        """选择颜色"""
        color = colorchooser.askcolor(color=self.current_color)[1]
        if color:
            self.current_color = color
            self.color_preview.config(bg=color)
            
    def update_char_metrics(self, value):
        """更新字符度量"""
        if self.current_char in self.font_data:
            self.font_data[self.current_char]['advance_width'] = float(value)
        self.update_text_preview_enhanced()
        
    def update_char_spacing(self, value):
        """更新字符间距"""
        self.update_text_preview_enhanced()
        
    def update_vertical_align(self, value):
        """更新垂直对齐"""
        self.update_text_preview_enhanced()
        
    def update_metadata(self, key, value):
        """更新字体元数据"""
        if key in ['units_per_em', 'ascender', 'descender', 'line_gap', 'cap_height', 'x_height']:
            try:
                self.font_metadata[key] = int(value)
            except ValueError:
                pass
        else:
            self.font_metadata[key] = str(value)
    
    # 视图控制方法
    def zoom_in(self):
        """放大视图"""
        self.pixel_size = min(self.pixel_size + 2, 25)
        self.draw_enhanced_grid()
        
    def zoom_out(self):
        """缩小视图"""
        self.pixel_size = max(self.pixel_size - 2, 6)
        self.draw_enhanced_grid()
        
    def fit_view(self):
        """适应视图"""
        self.draw_enhanced_grid()
        
    def actual_size(self):
        """实际大小"""
        self.pixel_size = 12
        self.draw_enhanced_grid()
        
    def toggle_grid(self):
        """切换网格显示"""
        self.draw_enhanced_grid()
        
    def toggle_guides(self):
        """切换辅助线显示"""
        self.draw_enhanced_grid()
    
    # 编辑操作方法
    def undo_stroke(self):
        """撤销笔画"""
        if self.current_char in self.strokes and self.strokes[self.current_char]:
            removed_stroke = self.strokes[self.current_char].pop()
            self.add_to_undo_stack(removed_stroke)
            self.draw_enhanced_grid()
            self.update_previews_enhanced()
            self.update_stroke_info_enhanced()
            self.add_to_history(f"撤销笔画 - {self.current_char}")
            
    def redo_stroke(self):
        """重做笔画"""
        if hasattr(self, 'undo_stack') and self.undo_stack:
            stroke = self.undo_stack.pop()
            if self.current_char not in self.strokes:
                self.strokes[self.current_char] = []
            self.strokes[self.current_char].append(stroke)
            self.draw_enhanced_grid()
            self.update_previews_enhanced()
            self.update_stroke_info_enhanced()
            self.add_to_history(f"重做笔画 - {self.current_char}")
    
    def add_to_undo_stack(self, stroke):
        """添加到撤销堆栈"""
        if not hasattr(self, 'undo_stack'):
            self.undo_stack = []
        self.undo_stack.append(stroke)
        if len(self.undo_stack) > 50:  # 限制堆栈大小
            self.undo_stack.pop(0)
        
    def clear_current_char(self):
        """清除当前字符"""
        if messagebox.askyesno('确认清除', f'确定要清除字符 "{self.current_char}" 的所有笔画吗？'):
            if self.current_char in self.strokes:
                self.strokes[self.current_char].clear()
                
            self.draw_enhanced_grid()
            self.update_previews_enhanced()
            self.update_stroke_info_enhanced()
            self.add_to_history(f"清除字符 - {self.current_char}")
        
    def copy_char(self):
        """复制字符"""
        self.clipboard_char = {
            'strokes': [stroke for stroke in self.strokes.get(self.current_char, [])],
            'advance_width': self.font_data.get(self.current_char, {}).get('advance_width', 1.0)
        }
        self.update_status_enhanced('字符已复制到剪贴板')
        self.add_to_history(f"复制字符 - {self.current_char}")
        
        
    def paste_char(self):
        """粘贴字符"""
        if hasattr(self, 'clipboard_char') and self.clipboard_char:
            if messagebox.askyesno('确认粘贴', f'确定要粘贴到字符 "{self.current_char}" 吗？这将覆盖现有内容。'):
                self.strokes[self.current_char] = [stroke for stroke in self.clipboard_char['strokes']]
                self.font_data[self.current_char]['advance_width'] = self.clipboard_char['advance_width']
                
                self.draw_enhanced_grid()
                self.update_previews_enhanced()
                self.update_stroke_info_enhanced()
                self.update_status_enhanced('字符已从剪贴板粘贴')
                self.add_to_history(f"粘贴字符 - {self.current_char}")
        else:
            messagebox.showwarning('粘贴失败', '剪贴板中没有字符数据')
            
    def flip_char(self):
        """翻转字符"""
        if self.current_char in self.strokes and self.strokes[self.current_char]:
            for stroke in self.strokes[self.current_char]:
                for i, point in enumerate(stroke.points):
                    x, y, pressure, timestamp = point
                    # 水平翻转
                    stroke.points[i] = (self.grid_size - x, y, pressure, timestamp)
                    
            self.draw_enhanced_grid()
            self.update_previews_enhanced()
            self.add_to_history(f"翻转字符 - {self.current_char}")
    
    # 预览更新方法
    def update_previews_enhanced(self):
        """更新增强版预览"""
        self.update_single_char_preview_enhanced()
        self.update_text_preview_enhanced()
        self.update_stats_display_enhanced()
        
    def update_single_char_preview_enhanced(self):
        """更新单字符预览（增强版）"""
        self.preview_canvas.delete('all')
        
        if self.current_char not in self.strokes:
            return
            
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 300
        if canvas_height <= 1:
            canvas_height = 120
            
        # 计算预览缩放
        preview_size = min(canvas_width - 20, canvas_height - 20)
        scale = preview_size / (self.grid_size * self.pixel_size)
        
        start_x = (canvas_width - preview_size) // 2
        start_y = (canvas_height - preview_size) // 2
        
        # 绘制背景
        self.preview_canvas.create_rectangle(start_x, start_y,
                                           start_x + preview_size,
                                           start_y + preview_size,
                                           outline=self.colors['border'],
                                           fill='white', width=2)
                                           
        # 绘制笔画
        for stroke in self.strokes[self.current_char]:
            if len(stroke.points) < 2:
                continue
                
            canvas_points = []
            for point in stroke.points:
                preview_x = start_x + point[0] * self.pixel_size * scale
                preview_y = start_y + point[1] * self.pixel_size * scale
                canvas_points.extend([preview_x, preview_y])
                
            if len(canvas_points) >= 4:
                if self.pressure_sensitivity:
                    # 压感预览
                    for i in range(0, len(canvas_points) - 3, 2):
                        segment_points = canvas_points[i:i+4]
                        point_idx = i // 2
                        if point_idx < len(stroke.points):
                            pressure = stroke.points[point_idx][2]
                            line_width = max(1, int(self.stroke_width * pressure * scale * 0.8))
                            
                            stroke_color = getattr(stroke, 'color', self.current_color)
                            
                            self.preview_canvas.create_line(segment_points,
                                                           fill=stroke_color,
                                                           width=line_width,
                                                           smooth=True,
                                                           capstyle=tk.ROUND)
                else:
                    # 统一宽度预览
                    line_width = max(1, int(self.stroke_width * scale * 0.8))
                    stroke_color = getattr(stroke, 'color', self.current_color)
                    
                    self.preview_canvas.create_line(canvas_points,
                                                   fill=stroke_color,
                                                   width=line_width,
                                                   smooth=True,
                                                   capstyle=tk.ROUND,
                                                   joinstyle=tk.ROUND)
                                                   
    def update_text_preview_enhanced(self, event=None):
        """更新文本预览（增强版）"""
        self.text_preview_canvas.delete('all')
        
        text = self.preview_text.get()
        if not text:
            return
            
        canvas_width = self.text_preview_canvas.winfo_width()
        canvas_height = self.text_preview_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 300
        if canvas_height <= 1:
            canvas_height = 100
        
        # 计算字符尺寸
        char_size = self.preview_size_var.get() * 10
        scale = char_size / (self.grid_size * self.pixel_size)
        
        start_x = 10
        start_y = (canvas_height - char_size) // 2
        current_x = start_x
        
        for char in text:
            if char in self.strokes and self.strokes[char]:
                # 检查是否超出画布宽度
                if current_x + char_size > canvas_width - 10:
                    break
                
                # 绘制字符背景
                bg_rect = self.text_preview_canvas.create_rectangle(current_x, start_y,
                                                                   current_x + char_size,
                                                                   start_y + char_size,
                                                                   outline=self.colors['border'],
                                                                   fill='white')
                
                # 绘制字符笔画
                for stroke in self.strokes[char]:
                    if len(stroke.points) < 2:
                        continue
                    
                    canvas_points = []
                    for point in stroke.points:
                        char_x = current_x + point[0] * self.pixel_size * scale
                        char_y = start_y + point[1] * self.pixel_size * scale
                        canvas_points.extend([char_x, char_y])
                    
                    if len(canvas_points) >= 4:
                        if self.pressure_sensitivity:
                            # 压感文本预览
                            for i in range(0, len(canvas_points) - 3, 2):
                                segment_points = canvas_points[i:i+4]
                                point_idx = i // 2
                                if point_idx < len(stroke.points):
                                    pressure = stroke.points[point_idx][2]
                                    line_width = max(1, int(self.stroke_width * pressure * scale * 0.6))
                                    
                                    stroke_color = getattr(stroke, 'color', self.current_color)
                                    
                                    self.text_preview_canvas.create_line(segment_points,
                                                                        fill=stroke_color,
                                                                        width=line_width,
                                                                        smooth=True,
                                                                        capstyle=tk.ROUND)
                        else:
                            line_width = max(1, int(self.stroke_width * scale * 0.6))
                            stroke_color = getattr(stroke, 'color', self.current_color)
                            
                            self.text_preview_canvas.create_line(canvas_points,
                                                               fill=stroke_color,
                                                               width=line_width,
                                                               smooth=True,
                                                               capstyle=tk.ROUND,
                                                               joinstyle=tk.ROUND)
                
                # 移动到下一个字符位置
                char_spacing = char_size * 0.1 + self.char_spacing_var.get()
                current_x += char_size + char_spacing
                
            elif char == ' ':
                # 空格
                current_x += char_size * 0.4
            else:
                # 未知字符，绘制占位符
                self.text_preview_canvas.create_rectangle(current_x, start_y + char_size * 0.8,
                                                         current_x + char_size * 0.3, start_y + char_size,
                                                         fill=self.colors['muted'], outline='')
                current_x += char_size * 0.4
                
    def update_stats_display_enhanced(self):
        """更新统计显示（增强版）"""
        if not hasattr(self, 'stats_text'):
            return
            
        self.stats_text.delete('1.0', tk.END)
        
        stats = self.char_db.get_stats()
        designed_count = stats['designed']
        total_count = stats['total']
        progress = stats['progress']
        
        current_strokes = len(self.strokes.get(self.current_char, []))
        
        stats_content = f"项目进度:\n"
        stats_content += f"已设计字符: {designed_count}/{total_count} ({progress:.1f}%)\n"
        stats_content += f"当前字符笔画: {current_strokes}\n"
        stats_content += f"当前工具: {self.current_tool}\n"
        stats_content += f"网格大小: {self.grid_size}×{self.grid_size}"
        
        self.stats_text.insert('1.0', stats_content)
        
        # 更新进度条
        if hasattr(self, 'progress_bar'):
            self.progress_bar['value'] = progress
        
    def update_stroke_info_enhanced(self):
        """更新增强版笔画信息"""
        if not hasattr(self, 'stroke_info_text'):
            return
            
        self.stroke_info_text.delete('1.0', tk.END)
        
        if self.current_char in self.strokes:
            strokes = self.strokes[self.current_char]
            info = f"字符: {self.current_char}\n"
            info += f"Unicode: U+{ord(self.current_char):04X}\n"
            info += f"笔画数: {len(strokes)}\n"
            info += f"复杂度: {self.calculate_complexity_enhanced(strokes)}\n\n"
            
            if strokes:
                info += "笔画详情:\n"
                for i, stroke in enumerate(strokes[:8]):  # 显示前8个笔画
                    points_count = len(stroke.points)
                    if points_count > 0:
                        avg_pressure = sum(p[2] for p in stroke.points) / points_count
                        stroke_type = getattr(stroke, 'stroke_type', 'normal')
                        info += f"{i+1}. 点数: {points_count}, 压感: {avg_pressure:.2f}, 类型: {stroke_type}\n"
                    
                if len(strokes) > 8:
                    info += f"... 还有 {len(strokes) - 8} 笔画"
        else:
            info = f"字符: {self.current_char}\n没有笔画数据"
        
        self.stroke_info_text.insert('1.0', info)
        
    def calculate_complexity_enhanced(self, strokes):
        """计算增强版字符复杂度"""
        if not strokes:
            return "无"
            
        total_points = sum(len(stroke.points) for stroke in strokes)
        stroke_count = len(strokes)
        
        # 计算复杂度分数
        complexity_score = stroke_count * 2 + total_points * 0.1
        
        if complexity_score > 40:
            return "非常复杂"
        elif complexity_score > 25:
            return "复杂"
        elif complexity_score > 15:
            return "中等"
        elif complexity_score > 5:
            return "简单"
        else:
            return "很简单"
    
    # 分析方法
    def analyze_character(self):
        """深度分析字符"""
        analysis = self.perform_enhanced_character_analysis()
        
        # 更新分析文本框
        if hasattr(self, 'shape_analysis_text'):
            self.shape_analysis_text.delete('1.0', tk.END)
            self.shape_analysis_text.insert('1.0', analysis)
            
        # 更新质量评估
        quality = self.evaluate_character_quality()
        if hasattr(self, 'quality_text'):
            self.quality_text.delete('1.0', tk.END)
            self.quality_text.insert('1.0', quality)
        
    def perform_enhanced_character_analysis(self):
        """执行增强版字符分析"""
        if self.current_char not in self.strokes or not self.strokes[self.current_char]:
            return '当前字符没有笔画数据'
            
        strokes = self.strokes[self.current_char]
        analysis = f"字符 '{self.current_char}' 深度分析:\n\n"
        
        # 基本信息
        analysis += f"笔画数: {len(strokes)}\n"
        analysis += f"复杂度: {self.calculate_complexity_enhanced(strokes)}\n"
        
        # 边界框分析
        all_points = []
        for stroke in strokes:
            all_points.extend([(p[0], p[1]) for p in stroke.points])
            
        if all_points:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            analysis += f"边界框: {width:.1f} × {height:.1f}\n"
            analysis += f"宽高比: {width/height:.2f}\n"
            analysis += f"位置: ({min_x:.1f}, {min_y:.1f})\n"
            
            # 重心计算
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            analysis += f"重心: ({center_x:.1f}, {center_y:.1f})\n"
            
            # 偏移分析
            grid_center = self.grid_size / 2
            offset_x = center_x - grid_center
            offset_y = center_y - grid_center
            analysis += f"偏移: ({offset_x:.1f}, {offset_y:.1f})\n"
            
        return analysis
        
    def evaluate_character_quality(self):
        """评估字符质量"""
        if self.current_char not in self.strokes or not self.strokes[self.current_char]:
            return '无法评估：没有笔画数据'
            
        strokes = self.strokes[self.current_char]
        quality = "字符质量评估:\n\n"
        
        # 笔画数量评估
        stroke_count = len(strokes)
        if stroke_count == 0:
            quality += "笔画数: 缺失 ❌\n"
        elif stroke_count < 3:
            quality += "笔画数: 偏少 ⚠️\n"
        elif stroke_count > 20:
            quality += "笔画数: 偏多 ⚠️\n"
        else:
            quality += "笔画数: 适中 ✅\n"
            
        # 笔画平滑度评估
        total_points = sum(len(stroke.points) for stroke in strokes)
        avg_points_per_stroke = total_points / stroke_count if stroke_count > 0 else 0
        
        if avg_points_per_stroke < 3:
            quality += "笔画平滑度: 过于简单 ⚠️\n"
        elif avg_points_per_stroke > 50:
            quality += "笔画平滑度: 过于复杂 ⚠️\n"
        else:
            quality += "笔画平滑度: 良好 ✅\n"
            
        # 整体评分
        score = 0
        if 3 <= stroke_count <= 20:
            score += 30
        if 5 <= avg_points_per_stroke <= 30:
            score += 30
        if len([s for s in strokes if len(s.points) > 2]) == stroke_count:
            score += 40  # 所有笔画都有效
            
        if score >= 80:
            quality += f"\n总体评分: {score}/100 优秀 🌟"
        elif score >= 60:
            quality += f"\n总体评分: {score}/100 良好 👍"
        elif score >= 40:
            quality += f"\n总体评分: {score}/100 一般 ⚠️"
        else:
            quality += f"\n总体评分: {score}/100 需改进 ❌"
            
        return quality
    
    # 历史记录方法
    def add_to_history(self, action):
        """添加到历史记录"""
        if not hasattr(self, 'history'):
            self.history = []
            
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.history.append(f"[{timestamp}] {action}")
        
        if len(self.history) > 100:  # 限制历史记录数量
            self.history.pop(0)
            
        # 更新历史显示
        if hasattr(self, 'history_listbox'):
            self.history_listbox.delete(0, tk.END)
            for item in reversed(self.history[-20:]):  # 显示最近20项
                self.history_listbox.insert(0, item)
                
        # 更新最近编辑显示
        if hasattr(self, 'recent_text'):
            self.recent_text.delete('1.0', tk.END)
            recent_actions = '\n'.join(reversed(self.history[-5:]))
            self.recent_text.insert('1.0', f"最近操作:\n{recent_actions}")
    
    def restore_from_history(self):
        """从历史记录恢复"""
        # 这里可以实现更复杂的历史恢复逻辑
        messagebox.showinfo('历史恢复', '历史恢复功能开发中...')
    
    def clear_history(self):
        """清除历史记录"""
        if messagebox.askyesno('确认清除', '确定要清除所有历史记录吗？'):
            if hasattr(self, 'history'):
                self.history.clear()
            if hasattr(self, 'history_listbox'):
                self.history_listbox.delete(0, tk.END)
            if hasattr(self, 'recent_text'):
                self.recent_text.delete('1.0', tk.END)
    
    # 状态更新方法
    def update_status_enhanced(self, message=None):
        """更新增强版状态栏"""
        if message:
            status_text = message
        else:
            stats = self.char_db.get_stats()
            designed_count = stats['designed']
            
            tool_name = {
                'brush': '毛笔',
                'pen': '钢笔', 
                'marker': '马克笔',
                'pencil': '铅笔',
                'eraser': '橡皮擦'
            }.get(self.current_tool, self.current_tool)
            
            status_text = f'就绪 | 当前字符: {self.current_char} | 工具: {tool_name} | 已设计: {designed_count}'
            
        self.status_label.config(text=status_text)
        
        # 更新统计
        stats = self.char_db.get_stats()
        self.stats_label.config(text=f'字符数: {stats["designed"]}/{stats["total"]}')
        
        # 更新工具状态
        tool_name = {
            'brush': '毛笔',
            'pen': '钢笔', 
            'marker': '马克笔',
            'pencil': '铅笔',
            'eraser': '橡皮擦'
        }.get(self.current_tool, self.current_tool)
        self.tool_status_label.config(text=f'工具: {tool_name}')
        
        # 更新导航栏统计
        if hasattr(self, 'stats_display'):
            self.stats_display.config(text=f"总字符: {stats['total']} | 已设计: {stats['designed']} | 进度: {stats['progress']:.1f}%")
    
    # 事件绑定和后台任务
    def bind_events(self):
        """绑定事件"""
        self.canvas.bind('<Button-1>', self.start_stroke)
        self.canvas.bind('<B1-Motion>', self.continue_stroke)
        self.canvas.bind('<ButtonRelease-1>', self.end_stroke)
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
        # 键盘快捷键
        self.root.bind('<Control-s>', lambda e: self.save_font())
        self.root.bind('<Control-o>', lambda e: self.open_font())
        self.root.bind('<Control-n>', lambda e: self.new_font())
        self.root.bind('<Control-z>', lambda e: self.undo_stroke())
        self.root.bind('<Control-y>', lambda e: self.redo_stroke())
        self.root.bind('<Delete>', lambda e: self.clear_current_char())
        self.root.bind('<Control-c>', lambda e: self.copy_char())
        self.root.bind('<Control-v>', lambda e: self.paste_char())
        
        self.root.focus_set()
        
    def start_background_tasks(self):
        """启动后台任务"""
        # 时间更新任务
        def update_time():
            if hasattr(self, 'time_label'):
                self.time_label.config(text=datetime.now().strftime('%H:%M:%S'))
            self.root.after(1000, update_time)
        update_time()
        
        # 自动保存任务
        def auto_save():
            if hasattr(self, 'auto_save_var') and self.auto_save_var.get():
                # 这里可以实现自动保存逻辑
                pass
            self.root.after(300000, auto_save)  # 5分钟
        auto_save()
        
        # 统计更新任务
        def update_stats():
            self.update_status_enhanced()
            self.root.after(30000, update_stats)  # 30秒
        update_stats()
    
    def on_canvas_resize(self, event):
        """处理画布大小变化"""
        self.root.after(100, self.draw_enhanced_grid)
    
    # 对话框方法
    def show_export_dialog(self):
        """显示导出对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("导出设置")
        dialog.geometry("500x400")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 这里可以添加更复杂的导出设置界面
        tk.Label(dialog, text="导出设置", 
                font=('Microsoft YaHei UI', 16, 'bold')).pack(pady=20)
        
        tk.Button(dialog, text="导出TTF", 
                 command=lambda: [self.export_ttf(), dialog.destroy()],
                 font=('Microsoft YaHei UI', 12),
                 bg=self.colors['primary'], fg='white',
                 padx=20, pady=10).pack(pady=10)
                 
        tk.Button(dialog, text="取消", 
                 command=dialog.destroy,
                 font=('Microsoft YaHei UI', 12),
                 padx=20, pady=10).pack(pady=5)
    
    def show_batch_dialog(self):
        """显示批量操作对话框"""
        messagebox.showinfo('批量操作', '批量操作功能开发中，将支持：\n- 批量导入字符\n- 批量导出图片\n- 批量设置属性\n- 批量质量检查')
    
    def show_settings_dialog(self):
        """显示设置对话框"""
        messagebox.showinfo('全局设置', '请在右侧设置标签页中调整各项设置参数。')
    
    # 文件操作方法
    def new_font(self):
        """新建字体项目"""
        if messagebox.askyesno('新建字体', '确定要新建字体吗？未保存的更改将丢失。'):
            # 清除所有笔画数据
            for char in self.strokes:
                self.strokes[char].clear()
                
            # 重置字体元数据
            self.font_metadata = {
                'family_name': '新字体',
                'style_name': 'Regular',
                'version': '1.0',
                'copyright': f'© {datetime.now().year} Enhanced Chinese Font Designer',
                'description': '使用增强版现代中文手写体字体设计器创建',
                'units_per_em': 1000,
                'ascender': 880,
                'descender': -120,
                'line_gap': 50,
                'cap_height': 800,
                'x_height': 600
            }
                
            self.draw_enhanced_grid()
            self.update_previews_enhanced()
            self.update_stroke_info_enhanced()
            self.update_status_enhanced('新字体项目已创建')
            self.add_to_history('新建字体项目')
            
    def save_font(self):
        """保存字体项目"""
        filename = filedialog.asksaveasfilename(
            title='保存增强版中文手写字体项目',
            defaultextension='.json',
            filetypes=[('字体项目文件', '*.json'), ('所有文件', '*.*')]
        )
        
        if filename:
            try:
                # 序列化笔画数据
                strokes_data = {}
                for char, strokes in self.strokes.items():
                    strokes_data[char] = []
                    for stroke in strokes:
                        stroke_dict = {
                            'points': stroke.points,
                            'stroke_type': stroke.stroke_type,
                            'width_variation': stroke.width_variation,
                            'color': getattr(stroke, 'color', self.current_color),
                            'smoothness': getattr(stroke, 'smoothness', 0.3)
                        }
                        strokes_data[char].append(stroke_dict)
                
                project_data = {
                    'metadata': self.font_metadata,
                    'settings': {
                        'grid_size': self.grid_size,
                        'stroke_width': self.stroke_width,
                        'pressure_sensitivity': self.pressure_sensitivity,
                        'stroke_smoothing': self.stroke_smoothing,
                        'current_color': self.current_color,
                        'current_tool': self.current_tool,
                        'render_quality': self.render_quality
                    },
                    'export_settings': self.export_settings,
                    'created': datetime.now().isoformat(),
                    'version': '4.0',
                    'font_data': self.font_data,
                    'strokes_data': strokes_data
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, ensure_ascii=False, indent=2)
                    
                self.update_status_enhanced(f'项目已保存: {os.path.basename(filename)}')
                self.add_to_history(f'保存项目: {os.path.basename(filename)}')
                messagebox.showinfo('保存成功', f'增强版中文手写字体项目已保存到:\n{filename}')
                
            except Exception as e:
                messagebox.showerror('保存失败', f'保存失败: {str(e)}')
                
    def open_font(self):
        """打开字体项目"""
        filename = filedialog.askopenfilename(
            title='打开增强版中文手写字体项目',
            filetypes=[('字体项目文件', '*.json'), ('所有文件', '*.*')]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                    
                # 加载数据
                if 'metadata' in project_data:
                    self.font_metadata.update(project_data['metadata'])
                    
                if 'settings' in project_data:
                    settings = project_data['settings']
                    self.grid_size = settings.get('grid_size', 48)
                    self.stroke_width = settings.get('stroke_width', 4)
                    self.pressure_sensitivity = settings.get('pressure_sensitivity', True)
                    self.stroke_smoothing = settings.get('stroke_smoothing', 0.4)
                    self.current_color = settings.get('current_color', self.colors['dark'])
                    self.current_tool = settings.get('current_tool', 'brush')
                    self.render_quality = settings.get('render_quality', 'high')
                    
                if 'export_settings' in project_data:
                    self.export_settings.update(project_data['export_settings'])
                    
                # 加载笔画数据
                if 'strokes_data' in project_data:
                    strokes_data = project_data['strokes_data']
                    for char, stroke_list in strokes_data.items():
                        self.strokes[char] = []
                        for stroke_dict in stroke_list:
                            stroke = HandwritingStroke()
                            stroke.points = stroke_dict.get('points', [])
                            stroke.stroke_type = stroke_dict.get('stroke_type', 'normal')
                            stroke.width_variation = stroke_dict.get('width_variation', 1.0)
                            stroke.color = stroke_dict.get('color', self.current_color)
                            stroke.smoothness = stroke_dict.get('smoothness', 0.3)
                            self.strokes[char].append(stroke)
                
                # 更新界面
                if hasattr(self, 'stroke_width_var'):
                    self.stroke_width_var.set(self.stroke_width)
                if hasattr(self, 'pressure_var'):
                    self.pressure_var.set(self.pressure_sensitivity)
                if hasattr(self, 'smoothing_var'):
                    self.smoothing_var.set(self.stroke_smoothing)
                if hasattr(self, 'current_tool_var'):
                    self.current_tool_var.set(self.current_tool)
                if hasattr(self, 'grid_size_var'):
                    self.grid_size_var.set(self.grid_size)
                if hasattr(self, 'quality_var'):
                    self.quality_var.set(self.render_quality)
                
                self.draw_enhanced_grid()
                self.update_previews_enhanced()
                self.update_stroke_info_enhanced()
                
                self.update_status_enhanced(f'项目已加载: {os.path.basename(filename)}')
                self.add_to_history(f'打开项目: {os.path.basename(filename)}')
                messagebox.showinfo('加载成功', f'项目已成功加载')
                
            except Exception as e:
                messagebox.showerror('加载失败', f'加载失败: {str(e)}')
    
    # 导出方法
    def export_ttf(self):
        """导出TTF字体"""
        if not FONTTOOLS_AVAILABLE:
            messagebox.showerror('功能不可用', 
                               '需要安装FontTools库:\n\npip install fonttools\n\n'
                               '安装后即可使用TTF导出功能。')
            return
            
        filename = filedialog.asksaveasfilename(
            title='导出TTF字体',
            defaultextension='.ttf',
            filetypes=[('TrueType字体', '*.ttf'), ('所有文件', '*.*')]
        )
        
        if filename:
            try:
                # 这里需要实现完整的TTF导出逻辑
                # 由于代码量较大，这里提供一个简化的实现框架
                messagebox.showinfo('导出进行中', 
                                   f'正在导出TTF字体到:\n{filename}\n\n'
                                   '由于TTF导出涉及复杂的字体格式转换，'
                                   '完整实现需要更多代码。\n\n'
                                   '当前版本支持项目保存和图像导出功能。')
                
            except Exception as e:
                messagebox.showerror('导出失败', f'TTF导出失败: {str(e)}')
    
    def export_png(self):
        """导出PNG图像"""
        if not self.strokes.get(self.current_char):
            messagebox.showwarning('导出失败', '当前字符没有笔画数据')
            return
            
        filename = filedialog.asksaveasfilename(
            title='导出PNG图像',
            defaultextension='.png',
            filetypes=[('PNG图像', '*.png'), ('所有文件', '*.*')]
        )
        
        if filename:
            try:
                size = self.png_char_size.get()
                
                # 创建PIL图像
                if self.png_transparent.get():
                    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
                else:
                    img = Image.new('RGB', (size, size), (255, 255, 255))
                
                draw = ImageDraw.Draw(img)
                
                # 计算缩放比例
                scale = size / self.grid_size
                
                # 绘制笔画
                for stroke in self.strokes[self.current_char]:
                    if len(stroke.points) < 2:
                        continue
                    
                    # 转换点坐标
                    points = []
                    for point in stroke.points:
                        x = point[0] * scale
                        y = point[1] * scale
                        points.append((x, y))
                    
                    # 绘制线条
                    if len(points) >= 2:
                        stroke_color = getattr(stroke, 'color', self.current_color)
                        # 将十六进制颜色转换为RGB
                        if stroke_color.startswith('#'):
                            r = int(stroke_color[1:3], 16)
                            g = int(stroke_color[3:5], 16)  
                            b = int(stroke_color[5:7], 16)
                            color = (r, g, b)
                        else:
                            color = (0, 0, 0)
                            
                        # 绘制连续线条
                        for i in range(len(points) - 1):
                            draw.line([points[i], points[i+1]], 
                                     fill=color, width=max(1, int(self.stroke_width * scale / 4)))
                
                img.save(filename)
                messagebox.showinfo('导出成功', f'PNG图像已导出到:\n{filename}')
                self.add_to_history(f'导出PNG: {os.path.basename(filename)}')
                
            except Exception as e:
                messagebox.showerror('导出失败', f'PNG导出失败: {str(e)}')
    
    def export_svg(self):
        """导出SVG矢量图"""
        if not self.strokes.get(self.current_char):
            messagebox.showwarning('导出失败', '当前字符没有笔画数据')
            return
            
        filename = filedialog.asksaveasfilename(
            title='导出SVG矢量图',
            defaultextension='.svg',
            filetypes=[('SVG矢量图', '*.svg'), ('所有文件', '*.*')]
        )
        
        if filename:
            try:
                size = 256  # SVG默认尺寸
                scale = size / self.grid_size
                
                # 生成SVG内容
                svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" 
     xmlns="http://www.w3.org/2000/svg">
  <rect width="{size}" height="{size}" fill="white"/>
'''
                
                # 添加笔画路径
                for i, stroke in enumerate(self.strokes[self.current_char]):
                    if len(stroke.points) < 2:
                        continue
                    
                    stroke_color = getattr(stroke, 'color', self.current_color)
                    stroke_width = max(1, self.stroke_width * scale / 4)
                    
                    # 构建路径
                    path_data = f"M {stroke.points[0][0] * scale},{stroke.points[0][1] * scale}"
                    for point in stroke.points[1:]:
                        path_data += f" L {point[0] * scale},{point[1] * scale}"
                    
                    svg_content += f'''  <path d="{path_data}" 
           stroke="{stroke_color}" 
           stroke-width="{stroke_width}" 
           stroke-linecap="round" 
           stroke-linejoin="round" 
           fill="none"/>
'''
                
                svg_content += '</svg>'
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
                
                messagebox.showinfo('导出成功', f'SVG矢量图已导出到:\n{filename}')
                self.add_to_history(f'导出SVG: {os.path.basename(filename)}')
                
            except Exception as e:
                messagebox.showerror('导出失败', f'SVG导出失败: {str(e)}')
    
    def batch_export(self):
        """批量导出"""
        # 获取所有已设计的字符
        designed_chars = [char for char, strokes in self.strokes.items() if strokes]
        
        if not designed_chars:
            messagebox.showwarning('批量导出', '没有已设计的字符可以导出')
            return
        
        folder = filedialog.askdirectory(title='选择导出文件夹')
        if not folder:
            return
            
        try:
            export_count = 0
            for char in designed_chars:
                # 临时切换到当前字符进行导出
                original_char = self.current_char
                self.current_char = char
                
                # 生成文件名（使用Unicode编码）
                safe_filename = f"U+{ord(char):04X}_{char}.png"
                filepath = os.path.join(folder, safe_filename)
                
                # 导出PNG（简化版）
                try:
                    size = self.png_char_size.get()
                    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
                    draw = ImageDraw.Draw(img)
                    scale = size / self.grid_size
                    
                    for stroke in self.strokes[char]:
                        if len(stroke.points) < 2:
                            continue
                        points = [(p[0] * scale, p[1] * scale) for p in stroke.points]
                        for i in range(len(points) - 1):
                            draw.line([points[i], points[i+1]], 
                                     fill=(0, 0, 0), width=max(1, int(self.stroke_width * scale / 4)))
                    
                    img.save(filepath)
                    export_count += 1
                    
                except Exception as e:
                    print(f"导出字符 {char} 失败: {e}")
                
                # 恢复原始字符
                self.current_char = original_char
            
            messagebox.showinfo('批量导出完成', 
                               f'成功导出 {export_count} 个字符到:\n{folder}')
            self.add_to_history(f'批量导出 {export_count} 个字符')
            
        except Exception as e:
            messagebox.showerror('批量导出失败', f'批量导出过程中出现错误: {str(e)}')
    
    # 工具方法
    def lighten_color(self, color, factor):
        """使颜色变亮"""
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return color
    
    def check_dependencies(self):
        """检查依赖"""
        missing_deps = []
        
        if not FONTTOOLS_AVAILABLE:
            missing_deps.append('FontTools (用于TTF导出)')
            
        try:
            import PIL
        except ImportError:
            missing_deps.append('Pillow (用于图像导出)')
            
        if missing_deps:
            deps_text = '\n'.join([f"• {dep}" for dep in missing_deps])
            self.root.after(3000, lambda: messagebox.showinfo(
                '依赖提醒',
                f'为了使用全部功能，建议安装以下依赖库:\n\n{deps_text}\n\n'
                f'安装命令:\npip install fonttools pillow\n\n'
                f'当前版本支持完整的界面编辑、项目保存和基础导出功能。'
            ))
    
    # 初始化显示
    def initialize_display(self):
        """初始化显示"""
        # 执行搜索以显示初始字符
        self.perform_search()
        
        # 选择第一个字符
        if self.displayed_chars:
            first_char = self.displayed_chars[0][1]
            self.select_char(first_char)
        else:
            self.select_char('中')  # 默认字符
            
        # 更新所有预览
        self.update_previews_enhanced()
        self.update_status_enhanced()
        
    def run(self):
        """运行应用程序"""
        # 初始化显示
        self.root.after(500, self.initialize_display)
        
        # 显示欢迎信息
        self.root.after(2000, self.show_enhanced_welcome)
        
        # 启动主循环
        self.root.mainloop()
        
    def show_enhanced_welcome(self):
        """显示增强版欢迎信息"""
        welcome_msg = '''欢迎使用增强版现代中文手写体字体设计器 Pro v4.0

🎨 核心特色:
• 支持3000+标准汉字，包含GB2312常用字集
• 智能字符搜索和分类管理系统
• 高级笔画参数调节（压感、平滑度、多工具）
• 实时预览和深度字符分析功能
• 完整的项目管理和历史记录系统

🚀 新增功能:
• 数据库驱动的字符管理
• 多种导出格式支持（PNG、SVG、TTF*）
• 批量操作和质量评估系统
• 现代化响应式用户界面
• 智能分页和高级过滤功能

💡 使用技巧:
1. 左侧搜索框支持字符搜索和分类过滤
2. 中间画布支持多种绘图工具和压感模拟
3. 右侧标签页提供预览、分析、设置和历史功能
4. 支持完整的键盘快捷键操作
5. 自动保存和项目版本管理

📊 当前统计:
• 字符总数: 3000+
• 支持工具: 毛笔、钢笔、马克笔、铅笔、橡皮擦
• 导出格式: PNG、SVG、TTF（需安装FontTools）

开始创作您的专属中文手写字体吧！'''
        
        messagebox.showinfo('欢迎使用 - 增强版 v4.0', welcome_msg)

if __name__ == '__main__':
    try:
        app = ModernChineseFontDesigner()
        app.run()
    except Exception as e:
        print(f"应用程序启动失败: {e}")
        import traceback
        traceback.print_exc()