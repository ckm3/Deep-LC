import numpy as np
from .config import INPUT_SIZE


lc_anchors_setting = (
    dict(
        layer="p3",
        stride=32,
        size=32,
        scale=np.linspace(1, 2, 3, endpoint=False),
    ),
    dict(
        layer="p4",
        stride=64,
        size=64,
        scale=np.linspace(1, 2, 3, endpoint=False),
    ),
    dict(
        layer="p5",
        stride=128,
        size=128,
        scale=np.linspace(1, 2, 3),
    ),
)

ps_anchors_setting = (
    dict(
    layer = "p1",
    column_size = INPUT_SIZE[1]//4,
    scale = 0.1,
    ),
    dict(
    layer = "p2",
    column_size = INPUT_SIZE[1]//8,
    scale = 0.2,
    ),
    dict(
    layer = "p3",
    column_size = INPUT_SIZE[1]//16,
    scale = 0.3,
    ),
    dict(
    layer = "p4",
    column_size = INPUT_SIZE[1]//32,
    scale = 0.4,
    )
)



def ps_anchors(anchors_setting=ps_anchors_setting):
    edge_anchors = np.zeros((0, 2), dtype=np.float32)
    for anchor_info in anchors_setting:
        middle_x_array = np.arange(0.5 / anchor_info["column_size"], 1.0, 1.0 / anchor_info["column_size"])
        center_anchor_map_template = np.zeros((anchor_info["column_size"], 2), dtype=np.float32)
        center_anchor_map_template[:, 0] = middle_x_array
        edge_anchor_map = np.zeros((anchor_info["column_size"], 2), dtype=np.float32)
        center_anchor_map = center_anchor_map_template.copy()
        edge_anchor_map[:, 0] = center_anchor_map[..., 0] - anchor_info["scale"] / 2.0
        edge_anchor_map[:, 1] = center_anchor_map[..., 0] + anchor_info["scale"] / 2.0
        edge_anchors = np.concatenate(
                (edge_anchors, edge_anchor_map.reshape(-1, 2))
                )
    return edge_anchors





def lc_anchors(anchors_setting=None, input_shape=INPUT_SIZE):
    """
    generate default anchor

    :param anchors_setting: all informations of anchors
    :param input_shape: shape of input images, e.g. (h, w)
    :return: center_anchors: # anchors * 4 (oy, ox, h, w)
             edge_anchors: # anchors * 4 (y0, x0, y1, x1)
             anchor_area: # anchors * 1 (area)
    """
    if anchors_setting is None:
        anchors_setting = lc_anchors_setting

    center_anchors = np.zeros((0, 2), dtype=np.float32)
    edge_anchors = np.zeros((0, 2), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:

        stride = anchor_info["stride"]
        size = anchor_info["size"]
        scales = anchor_info["scale"]
        output_shape = tuple((int(input_shape[1] / stride),) ) + (2,)
        ostart = stride / 2.0
        ox = np.arange(ostart, ostart + stride * output_shape[0], stride)
        ox = ox.reshape(output_shape[0])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, 0] = ox
        for scale in scales:
            center_anchor_map = center_anchor_map_template.copy()
            center_anchor_map[:, 1] = size * scale

            edge_anchor_map = np.concatenate(
                (
                    center_anchor_map[..., 0] - center_anchor_map[..., 1] / 2.0,
                    center_anchor_map[..., 0] + center_anchor_map[..., 1] / 2.0,
                ),
                axis=-1,
            )
            center_anchors = np.concatenate(
                (center_anchors, center_anchor_map.reshape(-1, 2))
            )
            edge_anchors = np.concatenate(
                (edge_anchors, edge_anchor_map.reshape(-1, 2))
            )

    return center_anchors, edge_anchors


def hard_nms_lc(cdds, topn=10, iou_thresh=0.25):
    if not (
        type(cdds).__module__ == "numpy" and len(cdds.shape) == 2 and cdds.shape[1] >= 1
    ):
        raise TypeError("edge_box_map should be N * 5+ ndarray")

    cdds = cdds.copy()
    indices = np.argsort(cdds[:, 0])
    cdds = cdds[indices]
    cdd_results = []

    res = cdds

    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == topn:
            return np.array(cdd_results)
        res = res[:-1]

        start_max = np.maximum(res[:, 1], cdd[1])
        end_min = np.minimum(res[:, 2], cdd[2])
        lengths = end_min - start_max
        intersec_map = lengths
        intersec_map[lengths < 0] = 0
        iou_map_cur = intersec_map / (
            (res[:, 2] - res[:, 1])
            + (cdd[2] - cdd[1])
            - intersec_map
        )
        res = res[iou_map_cur < iou_thresh]

    return np.array(cdd_results)


def hard_nms_ps(cdds, topn=10, iou_thresh=0.25):
    if not (
        type(cdds).__module__ == "numpy" and len(cdds.shape) == 2 and cdds.shape[1] >= 4
    ):
        raise TypeError("edge_box_map should be N * 3 ndarray")

    cdds = cdds.copy()

    # filter out cdds that are too long or too short
    # cdds = cdds[np.logical_and((cdds[:, 2] - cdds[:, 1]) >= ts_min_gap*10, (cdds[:, 2] - cdds[:, 1]) <= ts_length)]
    indices = np.argsort(cdds[:, 0])
    cdds = cdds[indices]
    cdd_results = []

    res = cdds

    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == topn:
            return np.array(cdd_results)
        res = res[:-1]

        start_max = np.maximum(res[:, 1], cdd[1])
        end_min = np.minimum(res[:, 2], cdd[2])
        lengths = end_min - start_max
        intersec_map = lengths
        intersec_map[lengths < 0] = 0
        iou_map_cur = intersec_map / (
            (res[:, 2] - res[:, 1])
            + (cdd[2] - cdd[1])
            - intersec_map
        )
        res = res[iou_map_cur < iou_thresh]

    return np.array(cdd_results)

