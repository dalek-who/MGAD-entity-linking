import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import copy


class RankHingeLoss(_Loss):
    """
    Creates a criterion that measures rank hinge loss.

    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """

    __constants__ = ['num_neg', 'margin', 'reduction']

    def __init__(self, num_neg: int, margin: float,
                 reduction: str = 'mean'):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        :param margin: Margin between positive and negative scores.
            Float. Has a default value of :math:`0`.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super().__init__()
        self.num_neg = num_neg
        self.margin = margin
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Calculate rank hinge loss.

        :param input: Predicted result.
        :param target: Label.
        :return: Hinge loss computed by user-defined margin.
        """
        # target: [1 0 0 0 1 0 0 0 1 0 0 0 ...], 1*pos_num, 0*neg_num follows a 1

        if input.ndim == 1:
            input = input[:, None]
            target = target.view_as(input)
        # input: [batch_size, 1]  batch_size = num_pos * (num_neg+1)
        # target: [batch_size, 1]

        assert target[::(self.num_neg + 1), :].cpu().numpy().sum() == target.cpu().numpy().sum() \
               and target[::(self.num_neg + 1), :].cpu().numpy().all(), target

        y_pos = input[::(self.num_neg + 1), :]
        # y_pos: [num_pos, 1]

        y_neg = []
        for neg_idx in range(self.num_neg):
            neg = input[(neg_idx + 1)::(self.num_neg + 1), :]
            # neg: [num_pos, 1]
            y_neg.append(neg)
        y_neg = torch.cat(y_neg, dim=-1)
        # y_neg: [num_pos, num_neg]

        y_neg = torch.mean(y_neg, dim=-1, keepdim=True)
        # y_neg: [num_pos, 1]

        target = torch.ones_like(y_pos)
        # y_pos: [num_pos, 1]

        return F.margin_ranking_loss(
            y_pos, y_neg, target,
            margin=self.margin,
            reduction=self.reduction
        )


class RankCrossEntropyLoss(_Loss):
    """Creates a criterion that measures rank cross entropy loss."""

    __constants__ = ['num_neg']

    def __init__(self, num_neg: int = 1):
        """
        :class:`RankCrossEntropyLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        """
        super().__init__()
        self.num_neg = num_neg

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Calculate rank cross entropy loss.

        :param input: Predicted result.
        :param target: Label.
        :return: Rank cross loss.
        """
        # target: [1 0 0 0 1 0 0 0 1 0 0 0 ...], 1*pos_num, 0*neg_num follows a 1

        if input.ndim == 1:
            input = input[:, None]
            target = target.view_as(input)
        # input: [batch_size, 1]  batch_size = num_pos * (num_neg+1)
        # target: [batch_size, 1]

        assert target[::(self.num_neg + 1), :].cpu().numpy().sum() == target.cpu().numpy().sum() \
               and target[::(self.num_neg + 1), :].cpu().numpy().all(), target

        logits = input[::(self.num_neg + 1), :]
        labels = target[::(self.num_neg + 1), :]
        # logits, labels: [batch_size, 1], batch_size = pos_num * (neg_num+1)
        for neg_idx in range(self.num_neg):
            neg_logits = input[(neg_idx + 1)::(self.num_neg + 1), :]
            neg_labels = target[(neg_idx + 1)::(self.num_neg + 1), :]
            # neg_logits, neg_labels: [pos_num, 1]
            logits = torch.cat((logits, neg_logits), dim=-1)
            labels = torch.cat((labels, neg_labels), dim=-1)
            # logits, labels: [pos_num, neg_idx+2], neg_id starts from 0
        # logits, labels: [pos_num, num_neg+1]

        # rank_cross_entropy = -logΣP(D+)，D+ is positive docs。apply sigmoid among doc scores to get P(D+), P(D-)
        return -torch.mean(
            torch.sum(
                labels * torch.log(F.softmax(logits, dim=-1) + torch.finfo(float).eps),  # add eps to avoid div-zero error
                dim=-1
            )
        )


class BCELoss_withClassWeight(_Loss):
    """
    给每一类设定不同权重的BCE Loss
    """
    __constants__ = ['reduction', 'class_weight']

    def __init__(self, class_weight: dict=None, size_average=None, reduce=None, reduction='mean'):
        """
        :param class_weight: 各类的权重字典例：{0:1, 1:2}
        :param size_average:
        :param reduce:
        :param reduction:
        """
        super(self.__class__, self).__init__(size_average, reduce, reduction)
        assert class_weight is None or isinstance(class_weight, dict)
        self.class_weight = copy.deepcopy(class_weight)

    def forward(self, input, target):
        weight = torch.ones_like(target)  # 把类别weight转换为样本weight
        if self.class_weight is not None:
            for c, w in self.class_weight.items():
                weight[target == c] = w
        return F.binary_cross_entropy(input=input, target=target.float(), weight=weight.float(), reduction=self.reduction)

if __name__=="__main__":
    x = torch.tensor([0.6, 0.7, 0.1])
    y = torch.tensor([1, -1, -1])

    my_loss = BCE_MarginRanking_Loss(margin=0.5, margin_loss_factor=1., reduction="mean")
    loss = my_loss(x, y)

    bce_loss = F.binary_cross_entropy(input=x, target=y.float(), reduction="mean")
    margin_loss = F.margin_ranking_loss(input1=x, input2=1-x, target=y.masked_fill(y == 0, -1), margin=0.5, reduction="mean")
    assert bce_loss + 1. * margin_loss == loss

    x = torch.tensor([0.9, 0.7, 0.1, 0.5, 0.5, 0.1, 0.2, 0.8])
    y = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    weight = torch.ones_like(y)
    weight[y == 1] = 2
    each_cross_entropy = y * torch.log(x) + (1 - y) * torch.log(1 - x)
    assert F.binary_cross_entropy(x, y.float(), weight=weight.float()) == -(each_cross_entropy * weight).mean()
    my_bce_loss_with_class_weight = BCELoss_withClassWeight(class_weight={0: 1, 1: 2})
    assert my_bce_loss_with_class_weight(x, y) == F.binary_cross_entropy(x, y.float(), weight=weight.float())
