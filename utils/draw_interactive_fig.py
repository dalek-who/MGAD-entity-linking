from typing import List, Dict, Tuple, NamedTuple, Union
from math import ceil, floor

import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, output_notebook, show, ColumnDataSource, save
from bokeh.models import CustomJS, HoverTool, Spinner, DataTable, TableColumn, Div, Button
from bokeh.plotting.figure import Figure as bokehFigure
from bokeh.layouts import row, column, LayoutDOM
from bokeh.transform import factor_cmap, factor_mark
import bokeh.events as bokeh_events


def interact_fig_score_label_scatter(
        label_true: np.ndarray,
        score_pred: np.ndarray,
        example_id: np.ndarray
) -> Union[bokehFigure, LayoutDOM]:
    """
    交互式的score散点图，便于分析正负样本的打分分布

    :param label_true:
    :param score_pred:
    :param example_id:
    :return:
    """
    data_num = len(label_true)
    radius = data_num / 500  # 每个点的半径

    idx = np.arange(len(label_true))
    color_map = {0: "blue", 1: "red"}  # 类别->颜色的映射
    assert set(color_map.keys()) == set(label_true)

    # 交互图表上的工具
    TOOLS = [
        "crosshair",  # 鼠标在图上的十字定位
        "pan",  # 可拖动图片
        "wheel_zoom",  # 鼠标滚轮缩放
        "box_zoom",  # 拉方形区域缩放
        "reset",  # 重置之前选的点
        "tap",  # 单击选择单个点，按shift可选多个点
        "box_select",  # 方块选择部分点
        "lasso_select",  # 套索选择部分点
        "poly_select",  # 多边形选择工具，鼠标左击定下多边形的顶点，圈定其中的范围
        "save",  # 保存图像截图
    ]

    # cosine score_pred range is [-1, 1], but label_true may be [0,1]
    y_min = min(label_true.min(), floor(score_pred.min()))
    y_max = max(label_true.max(), ceil(score_pred.max()))

    fig = figure(
        # 交互图的基本工具
        tools=TOOLS,
        # 图表title
        title="Predict Score Scatter",
        # 坐标范围
        x_range=(0, len(label_true)),
        y_range=(y_min, y_max),
        # 坐标轴名字
        x_axis_label='index',
        y_axis_label='Predict Score',
    )

    # 鼠标悬浮显示工具及显示内容
    hover = HoverTool(
        tooltips=[
            # ("index", "$index"),  # $index: 内部绝对index
            # ("(x,y)", "($x, $y)"),  # $x, $y: 鼠标位置坐标
            ("example_id", "@example_id"),  # @开头的：自定义的域
            ("label", "@label_true"),
            ("score_pred", "@score_pred"),
        ]
    )

    # 添加悬浮显示工具
    fig.add_tools(hover)

    # 添加由事件触发的交互框
    width_spiner = 80
    spinner_radius = Spinner(title="Radius", step=1, value=radius)  # 用来设置点的半径
    spinner_alpha = Spinner(title="Alpha", step=0.05, value=0.3)  # 用来设置点的透明度

    # 按不同label依次添加相应数据，为该label自动添加图例，并且可交互式隐藏相应label数据
    scatter_source_list: List[ColumnDataSource] = []  # 每个label的data_source
    each_label_data_num: List[int] = []  # 每个label有多少数据
    for label_i in color_map.keys():
        label_i_index = (label_true == label_i)
        # ColumnDataSource是动态画图的核心部件之一
        # 作为画图的数据源，ColumnDataSource是动态画图的核心部件之一
        # ColumnDataSource的数据值更改可以触发事件，且利用callback更改其他ColumnDataSource，实时更新到图上
        scatter_source = ColumnDataSource(data={
            "idx": idx[label_i_index],
            "score_pred": score_pred[label_i_index],
            "label_true": label_true[label_i_index],
            "example_id": example_id[label_i_index],
        })
        scatter_source_list.append(scatter_source)
        each_label_data_num.append(sum(label_i_index))

        circle = fig.circle(
            source=scatter_source,
            x='idx',
            y='score_pred',
            color=color_map[label_i],
            radius=radius,
            fill_alpha=0.3,
            line_color=None,
            legend_label=f"label_{label_i}",
        )
        spinner_radius.js_link('value', circle.glyph, 'radius')  # 将交互框的值与点的半径绑定
        spinner_alpha.js_link("value", circle.glyph, "fill_alpha")  # 将交互框的值与点的透明度绑定

    # 点图例后会发生什么
    fig.legend.click_policy = "hide"  # hide: 该图例的数据隐藏, mute: 该图例的数据变灰

    # 显示选中的点的数量
    width_div, height_div = 320, 10
    # Div组件，可用来显示html格式文本，普通文本也可以
    list_div_select_data_num = [
        Div(text=f"Selected label_{i}: 0", style={'font-size': '200%', 'color': color_map[i]})
        for i in range(len(each_label_data_num))
    ]
    for i in range(len(each_label_data_num)):
        div = list_div_select_data_num[i]
        source = scatter_source_list[i]
        callback_select = CustomJS(
            args={'div': div, 'source': source},  # args即下面的js code里的参数
            code=f"""
                div.text = 'Selected label_{i}: ' + source.selected.indices.length;
                """
        )
        source.selected.js_on_change('indices', callback_select)

    # 清空选择
    # 按钮组件
    button_clear_selection = Button(label='Clear selection', button_type='success')
    for i in range(len(each_label_data_num)):
        source = scatter_source_list[i]
        callback_click = CustomJS(
            args={'source': source},
            code="""
                source.selected.indices = [];
            """
        )
        button_clear_selection.js_on_event(bokeh_events.ButtonClick, callback_click)

    # 布局，columns表示纵向排列，row表示横向排列。row、column可嵌套
    layout = row(column(spinner_radius, spinner_alpha), column(fig, *list_div_select_data_num, button_clear_selection))
    """
    layout:
     
    spinner_radius |  fig
    spinner_alpha  |  div[0]
                   |  div[1]
                   |   ...
                   |  button_clear_selection
    """
    return layout


if __name__ == '__main__':
    import numpy as np

    seed = 42
    data_num = 10000

    np.random.seed(seed)

    label_true = np.random.randint(0, 2, data_num)
    score_pred = label_true \
                 - ((label_true == 1) * (np.random.rand(data_num) / 1.7)) \
                 + ((label_true == 0) * (np.random.rand(data_num) / 1.7))

    example_id = np.array([f"query_{i // 2}~{label_true[i]}" for i in range(data_num)])

    p = interact_fig_score_label_scatter(label_true=label_true, score_pred=score_pred, example_id=example_id)
    # save
    output_file("active_score_scatter.html")
    save(p)

