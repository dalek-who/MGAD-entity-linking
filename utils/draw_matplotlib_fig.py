from collections import Counter
from math import ceil, floor

import numpy as np
from matplotlib import pyplot as plt, ticker as ticker, patches as mpatches
from matplotlib.figure import Figure


def fig_confusion_matrix(confusion, categories_list) -> Figure:
    """
    matplotlib画混淆矩阵
    :param confusion: 混淆矩阵
    :param categories_list: 混淆矩阵的类别名字list
    :return: fig
    """
    # 绘制混淆矩阵。对每类的分类情况做了归一化，防止有的类数量太多颜色太深，其他类数量少颜色太浅
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    normalize_confusion = confusion / confusion.sum(1).reshape((-1,1))  # 混淆矩阵沿行归一化
    ax_confusion_matrix = ax.matshow(normalize_confusion)  # 混淆矩阵可视化
    fig.colorbar(ax_confusion_matrix)  # 添加颜色条

    # Set up axes
    ax.set_xticklabels([''] + categories_list, rotation=90)  # x轴标签：类别名称。rotation: 文字旋转90度，立起来，这样才能排的开
    list_categories_and_num = [f"{num}  {c}" for num, c in zip(confusion.sum(1), categories_list)]  # y轴标签：类别样本数量+类别名称
    ax.set_yticklabels([''] + list_categories_and_num)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # x主刻度线
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # y主刻度线

    fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
    return fig


def fig_images(img) -> Figure:
    """
    matplotlib绘制图片（image）
    :param img: image
    :return: fig
    """
    fig, ax = plt.subplots()
    npimg = img.numpy()
    plt_img = ax.imshow(np.transpose(npimg, (1, 2, 0)))
    fig.tight_layout()
    return fig


def fig_loss_and_lr_together(
        global_step: list or np.ndarray,
        step_train_loss: list or np.ndarray,
        step_lr: list or np.ndarray,
        evaluation_global_step: list or np.ndarray,
        evaluation_train_loss: list or np.ndarray,
        evaluation_valid_loss: list or np.ndarray
) -> Figure:
    """
    把train_step_loss, lr, evaluation_train_loss, evaluation_valid_loss几条曲线画在一张图上
    :param global_step: global step
    :param step_train_loss: 每次更新时的step loss
    :param step_lr: 每个step的学习率
    :param evaluation_global_step: 进行evaluation的global step list
    :param evaluation_train_loss: 每次evaluation的train loss
    :param evaluation_valid_loss: 每次evaluation的valid loss
    :return: fig
    """
    global_step = np.array(global_step)
    step_train_loss = np.array(step_train_loss)
    step_lr = np.array(step_lr)
    evaluation_global_step = np.array(evaluation_global_step)
    evaluation_train_loss = np.array(evaluation_train_loss)
    evaluation_valid_loss = np.array(evaluation_valid_loss)

    # epoch和step的loss图
    fig, ax_step_train_loss = plt.subplots()
    plot_step_train_loss = ax_step_train_loss.plot(global_step, step_train_loss, label="step train loss")
    plots = plot_step_train_loss
    if evaluation_train_loss.size != 0:
        plot_evaluation_train_loss = ax_step_train_loss.plot(evaluation_global_step, evaluation_train_loss, label="evaluation train loss")
        plots += plot_evaluation_train_loss
    plot_evaluation_valid_loss = ax_step_train_loss.plot(evaluation_global_step, evaluation_valid_loss, label="evaluation valid loss")
    plots += plot_evaluation_valid_loss

    ax_step_train_loss.set_xlabel("global step")  # x轴的标签
    ax_step_train_loss.set_ylabel("loss")  # y轴的标签
    # ax_step_train_loss.legend()  # 添加图例  # 后面把多个ax的图例统一添加到一起
    ax_step_train_loss.grid()  # 背景显示网格线

    # 叠加一个新的图用来标evaluation global step x轴刻度
    # evaluation loss不要画在这个子图上，否则图例与train loss的不在一个方格里
    ax_evaluation_train_loss = ax_step_train_loss.twiny()  # 叠加在原ax_step_train_loss图上，共享y轴
    ax_evaluation_train_loss.set_xlim(ax_step_train_loss.get_xlim())  # x轴对齐
    ax_evaluation_train_loss.set_xticks(evaluation_global_step[::5])  # 画evaluation global step刻度线和数值
    ax_evaluation_train_loss.set_xlabel("Evaluation global step")  # x轴的标签

    # 用☆标注上evaluation valid loss最小的位置
    argmin_eval_valid_loss = evaluation_valid_loss.argmin()
    min_eval_valid_loss_x, min_eval_valid_loss_y = evaluation_global_step[argmin_eval_valid_loss], evaluation_valid_loss[argmin_eval_valid_loss]
    ax_evaluation_train_loss.scatter([min_eval_valid_loss_x], [min_eval_valid_loss_y], color="black", marker="*")  # 添加eval valid loss最小的点

    # lr图
    ax_step_lr = ax_step_train_loss.twinx()  # 叠加在原ax_loss_step图上，共享x轴
    plot_step_lr = ax_step_lr.plot(global_step, step_lr, color="red", label="lr")
    plots += plot_step_lr
    ax_step_lr.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))  # y轴数字用保留两位小数的科学技术法表示
    ax_step_lr.set_ylabel("learning rate")  # y轴标签
    # ax_step_lr.legend(["lr"])  # 添加图例  # 后面把多个ax的图例统一添加到一起

    # 三个图例画在一起。否则每个ax会单独生成一个图例
    # plots = plot_step_train_loss + plot_evaluation_train_loss + plot_evaluation_valid_loss + plot_step_lr
    labels = [p.get_label() for p in plots]
    ax_step_train_loss.legend(plots, labels)

    # 保存与显示
    fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下

    return fig


def fig_find_lr_plot(lrs: list, losses: list) -> Figure:
    """
    寻找最佳学习率的曲线图
    :param lrs: 按指数增加的学习率变化图
    :param losses: loss变化图
    :return: fig
    """
    lrs, losses = np.array(lrs), np.array(losses)
    fig, ax_loss = plt.subplots()
    plot_loss = ax_loss.plot(np.log10(lrs)[10:-5], losses[10:-5], label="loss", color="blue")
    ax_loss.set_xlabel('log10 lr')  # x轴title
    ax_loss.set_ylabel("loss")  # y轴title
    # ax_loss.legend()  # 图例
    ax_loss.set_title("find learning rate")  # y可以调整标题的位置

    # lr图
    ax_log10_loss = ax_loss.twinx()  # 叠加在原ax_loss图上，共享x轴
    plot_log10_loss = ax_log10_loss.plot(np.log10(lrs)[10:-5], np.log10(losses)[10:-5], label="log10 loss", color="red")
    ax_log10_loss.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))  # y轴数字用保留两位小数的科学技术法表示
    ax_log10_loss.set_ylabel("log10 loss")  # y轴标签
    # ax_log10_loss.legend()  # 添加图例

    # 图例画在一起。否则每个ax会单独生成一个图例
    plots = plot_loss + plot_log10_loss
    labels = [p.get_label() for p in plots]
    ax_loss.legend(plots, labels)

    # 保存与显示
    fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下

    return fig


def fig_score_label_scatter(label_truth: np.ndarray, score_pred: np.ndarray) -> Figure:
    """
    score与label的分布散点图，每个score一个点，每种颜色代表一个label
    :param label_truth: ground truth label
    :param score_pred: 预测的score
    :return: fig
    """
    label_2_color = {0: "blue", 1: "red"}  # 类别->颜色的映射
    colors = [label_2_color[label] for label in label_truth]  # 每个点的颜色，用以区分不同的类别
    x = np.arange(len(score_pred))  # 每个点一个index

    fig, ax = plt.subplots()
    # 散点图，两个一起画。如果一个label一个label的话，后一个颜色可能会太浓密遮住前一个颜色
    ax.scatter(x=x, y=score_pred, c=colors, alpha=0.3, s=1)  # alpha透明度，透明的更容易看出密度；s是点的size

    # 手动设置图例
    handles=[mpatches.Patch(color=color, label=f'label {label}') for label, color in label_2_color.items()]
    ax.legend(handles=handles)

    ax.set_xlabel("example")  # x标签
    ax.set_ylabel("score")  # y标签

    # 设置y轴上下限
    y_min = min(label_truth.min(), floor(score_pred.min()))
    y_max = max(label_truth.max(), ceil(score_pred.max()))
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
    return fig


def fig_score_label_distribution(label_truth: np.ndarray, score_pred: np.ndarray, normalize: bool) -> Figure:
    """
    画每种label按score的分布情况，每个label有一个分布曲线
    :param label_truth: ground truth label
    :param score_pred: 预测的score
    :param normalize: 是否对点的数量归一化
    :return: fig
    """
    fig, ax = plt.subplots()

    for label in np.unique(label_truth):
        score_i = score_pred[label_truth == label]  # 真实标签为label的所有点的score
        ct = Counter(score_i.round(2))  # 对标签为label的所有点，先保留两位小数，然后统计每个score的点的数量
        sorted_score_i = {score: ct[score] for score in sorted(ct.keys())}   # score排序
        x = np.array(list(sorted_score_i.keys()))  # x：score
        y = np.array(list(sorted_score_i.values()))  # y：点的数量
        if normalize:
            y = y / y.sum()  # y: 点的数量归一化
        ax.plot(x, y, label=f"data with label {label}")
    ax.set_xlabel("score")
    ylabel = "log percent" if normalize else "log number"
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")  # y轴标度log化，否则数量级差异过大
    ax.legend()  # 图例
    fig.tight_layout()   # 适应窗口大小，否则可能有东西在窗口里画不下
    return fig