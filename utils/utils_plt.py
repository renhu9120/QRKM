import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

# Line styles accepted as the ``linestyle=`` keyword alone (not combined with a marker).
_PURE_LINE_STYLES = frozenset({
    "-",
    "--",
    "-.",
    ":",
    "None",
    " ",
    "",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
})

def plot_conv_curvs(x, data_series, title="", xlabel="Iteration", ylabel="log(dist(x,z))", figsize=(10, 6), grid=True,
                    filename="", yscale: str = "linear", y_log_floor: float = 1e-300,
                    show_preview: bool = False, legend_loc: str = "best",
                    legend_fontsize: float | str | None = None):
    """
    将多组向量绘制在同一张图中。

    参数：
    - x: 横坐标向量（与各条曲线共用时的默认值）
    - data_series: 列表，元素为字典，格式如：
        {"y": y1, "label": "method1", "style": "-", "color": "b"}
      若某项包含 ``"x"``，该条曲线使用该横轴；否则使用上面的共用 ``x``。
      可选 ``linestyle``（优先于 ``style``）、``marker``、``markersize``、``markevery``；
      若 ``linestyle`` 为组合格式串（如 ``"-*"``、``"-o"``、``"--x"``），按 ``plt.plot(x, y, fmt, ...)`` 绘制；
      纯线型（``"-"``、``"--"`` 等）仍走 ``linestyle=`` / ``marker=`` 参数路径。
      若给出 ``marker`` 且未给 ``markevery``，则按横轴长度自动稀疏取点。
    - title: 图标题
    - xlabel, ylabel: 坐标轴名称
    - figsize: 图像大小
    - grid: 是否显示网格
    - save_path: 若提供文件名，则保存图像到该路径
    - yscale: ``"linear"``（默认）或 ``"log"``；对数纵轴时用 LaTeX 风格主刻度
      ``$10^{-1}$, $10^{-2}$, …``（``LogFormatterMathtext``）
    - y_log_floor: ``yscale="log"`` 时将 ``y`` 裁剪到下界以上，避免非正数
    - show_preview: 为 True 且指定 ``filename`` 时，在写入 PDF 之后调用 ``plt.show()`` 弹出窗口预览
      （与已保存的图一致），关闭窗口后 ``close``（默认 False，避免批处理脚本被阻塞）
    - legend_loc: 图例位置（例如 ``"upper right"``, ``"lower left"``, ``"best"``）
    - legend_fontsize: 图例字号（如 ``10``、``"small"``）；None 表示使用 matplotlib 默认值

    用法示例：
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    data_series = [
        {"y": y1, "label": "sin(x)", "style": "-", "color": "blue"},
        {"y": y2, "label": "cos(x)", "style": "--", "color": "red"}
    ]
    plot_vectors(x, data_series)
    """

    plt.figure(figsize=figsize)

    for series in data_series:
        y = series["y"]
        label = series.get("label", "")
        linestyle = str(series.get("linestyle", series.get("style", "-"))).strip()
        marker = series.get("marker", None)
        markersize = series.get("markersize", 5.0)
        color = series.get("color", None)
        markevery = series.get("markevery", None)
        y_plot = y
        if yscale == "log":
            y_plot = np.maximum(np.asarray(y, dtype=float), float(y_log_floor))
        x_src = series.get("x", x)
        x_arr = np.asarray(x_src, dtype=float).ravel()
        if marker is not None and markevery is None:
            markevery = max(1, int(len(x_arr)) // 25) if len(x_arr) > 0 else None
        plot_kw: dict = {
            "label": label,
            "color": color,
            "linewidth": 2,
            "markersize": markersize,
        }
        if markevery is not None:
            plot_kw["markevery"] = markevery
        use_pure_ls = linestyle in _PURE_LINE_STYLES or marker is not None
        if use_pure_ls:
            plot_kw["linestyle"] = linestyle
            if marker is not None:
                plot_kw["marker"] = marker
            plt.plot(x_arr, y_plot, **plot_kw)
        else:
            # Matplotlib format code, e.g. ``"-*"``, ``"--o"`` (invalid as ``linestyle=`` alone).
            plt.plot(x_arr, y_plot, linestyle, **plot_kw)

    ax = plt.gca()
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    if grid:
        plt.grid(True, which="major", linestyle="--", alpha=0.45)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    # plt.legend(loc="upper right")  # 与plt.legend(loc=1)等价
    plt.tight_layout()

    # 判断 filename 是否非空，如果是就保存图片
    if filename.strip():
        path = f"{filename}.pdf"
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(path, format="pdf", bbox_inches="tight")
        print(f"Figure saved as {path}")
        if show_preview:
            plt.show()
        plt.close()
    else:
        plt.show()



def plot_sr_curvs(
        alg_list,
        *,
        x_key: str,
        y_key: str,
        xlabel: str,
        ylabel: str,
        title: str = "",
        figsize=(10, 6),
        save_name: str = "comparison_plot.pdf",
        dpi: int = 300
):
    """
    Plot comparison curves for multiple algorithms using their CSV data files.

    Parameters
    ----------
    alg_list : list of dict
        Each element is:
        {
            "file": "filename.csv",
            "label": "Algorithm Name",
            "linestyle": "--",
            "color": "red"
        }

    x_key : str
        Column name for x-axis (e.g., 'nd_rate').

    y_key : str
        Column name for y-axis (e.g., 'success_rate', 'avg_iters').

    xlabel, ylabel : str
        Axis labels for the plot.

    title : str, optional
        Title of the plot.

    figsize : tuple
        Figure size.

    save_name : str
        PDF file name to save the figure.

    dpi : int
        DPI for high-quality paper figures.
    """
    plt.figure(figsize=figsize)

    for item in alg_list:
        filename = item["file"]
        label = item.get("label", "Algorithm")
        ls = item.get("linestyle", "-")
        color = item.get("color", None)

        # Read CSV
        df = pd.read_csv(filename)

        # Ensure x and y keys exist
        if x_key not in df.columns or y_key not in df.columns:
            raise KeyError(f"Column {x_key} or {y_key} not found in file {filename}")

        x = df[x_key].values
        y = df[y_key].values

        # Plot
        plt.plot(
            x, y,
            ls,
            color=color,
            linewidth=2,
            markersize=5,
            label=label
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if title:
        plt.title(title, fontsize=13)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.legend(loc="upper left")  # 与plt.legend(loc=1)等价

    # 判断 filename 是否非空，如果是就保存图片
    if save_name.strip():
        # Save figure
        plt.tight_layout()
        plt.savefig(save_name, dpi=dpi)

    plt.show()
    print(f"Figure saved as {save_name}")
