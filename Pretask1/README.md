# 2048 HTML 小游戏

一个使用原生 **HTML + CSS + JavaScript** 实现的简单 2048 小游戏。

## 文件结构

```text
Pretask1/
├── index.html
├── style.css
├── script.js
└── README.md
```

## 运行方式

### 方式一：直接打开

直接用浏览器打开 `index.html` 即可运行：

```bash
xdg-open index.html
```

也可以在文件管理器中双击 `index.html`。

### 方式二：启动本地静态服务器（推荐）

在 `Pretask1/` 目录下执行：

```bash
python3 -m http.server 8000
```

然后在浏览器访问：

```text
http://localhost:8000
```

## 游戏说明

- 使用键盘 **方向键** 或 **W/A/S/D** 控制移动。
- 两个相同数字方块碰撞后会合并。
- 合成 `2048` 即视为达成目标。
- 当棋盘无法继续移动时，游戏结束。
- 点击“新游戏”按钮可重新开始。

## 特性

- 4×4 标准棋盘
- 实时得分与最高分显示
- 合成 2048 提示
- 游戏结束提示
