from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .resnet import resnet18
import numpy as np
from .anchors import (
    lc_anchors,
    hard_nms_lc,
    ps_anchors,
    hard_nms_ps,
)
from .config import CAT_NUM, PROPOSAL_NUM
from .dataset import lc_to_image_array, plot_phase_curve


class LC_ProposalNet(nn.Module):
    def __init__(self):
        super(LC_ProposalNet, self).__init__()
        self.expansion = 1
        self.down1 = nn.Conv2d(512 * self.expansion, 256, 3, 1, 1)
        self.down2 = nn.Conv2d(256, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(256, 3, kernel_size=(8, 1))
        self.tidy2 = nn.Conv2d(128, 3, kernel_size=(4, 1))
        self.tidy3 = nn.Conv2d(128, 3, kernel_size=(2, 1))

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class PS_ProposalNet(nn.Module):
    def __init__(self):
        super(PS_ProposalNet, self).__init__()
        self.expansion = 1
        self.smooth1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.smooth2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.smooth3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 1, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 1, 1, 1, 0)
        self.tidy4 = nn.Conv2d(512, 1, 1, 1, 0)
        self.pl1 = nn.Conv2d(1, 1, kernel_size=(64, 1))
        self.pl2 = nn.Conv2d(1, 1, kernel_size=(32, 1))
        self.pl3 = nn.Conv2d(1, 1, kernel_size=(16, 1))
        self.pl4 = nn.Conv2d(1, 1, kernel_size=(8, 1))

    def forward(self, p2, p3, p4, p5):
        batch_size = p2.size(0)
        p2 = self.ReLU(self.smooth1(p2))  # ([1, 128, 112, 112])
        p3 = self.ReLU(self.smooth2(p3))  # ([1, 128, 56, 56])
        p4 = self.ReLU(self.smooth3(p4))  # ([1, 128, 28, 28])
        t1 = self.tidy1(p2)  # torch.Size([1, 3, 112, 112])
        t2 = self.tidy2(p3)  # torch.Size([1, 3, 56, 56])
        t3 = self.tidy3(p4)  # torch.Size([1, 3, 28, 28])
        t4 = self.tidy4(p5)  # torch.Size([1, 2, 14, 14])
        col_scores_1 = self.pl1(t1).view(batch_size, -1)
        col_scores_2 = self.pl2(t2).view(batch_size, -1)
        col_scores_3 = self.pl3(t3).view(batch_size, -1)
        col_scores_4 = self.pl4(t4).view(batch_size, -1)
        return torch.cat(
            (col_scores_1, col_scores_2, col_scores_3, col_scores_4), dim=1
        )


class lc_component(nn.Module):
    def __init__(self, topN=4, nclasses=7, device="cuda"):
        super(lc_component, self).__init__()
        self.device = device
        self.pretrained_model = resnet18(pretrained=True).to(self.device)
        self.expansion = 1
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.proposal_net = LC_ProposalNet()
        self.fc = nn.Sequential(
            nn.Linear(3 + 512, 256), nn.ReLU(inplace=True), nn.Linear(256, nclasses)
        )
        self.topN = topN
        self.concat_net = nn.Sequential(
            nn.Linear((512 * self.expansion + 3) * (CAT_NUM + 1), nclasses),
        )

        self.partcls_net = nn.Sequential(
            nn.Linear(512 * self.expansion + 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, nclasses),
        )
        _, edge_anchors = lc_anchors()
        self.edge_anchors = edge_anchors.astype(int)

    def forward(
        self,
        lc_img,
        ps_img,
        phase_img,
        lc_params,
        ps_params,
        lc_list,
        ps_list,
        return_part_imgs=False,
        return_part_data=False,
        combined_mode=False,
    ):
        lc_img = lc_img.float()
        _, _, _, rpn_feature, feature = self.pretrained_model(lc_img)
        batch = lc_img.size(0)
        lc_feature_out = torch.cat([feature, lc_params], dim=1)
        resnet_out = self.fc(lc_feature_out)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate(
                (
                    x.reshape(-1, 1),
                    self.edge_anchors.copy(),
                    np.arange(0, len(x)).reshape(-1, 1),
                ),
                axis=1,
            )
            for x in rpn_score.data.cpu().numpy()
        ]
        top_n_cdds = [hard_nms_lc(x, topn=self.topN, iou_thresh=0.5) for x in all_cdds]
        top_n_cdds = np.concatenate(top_n_cdds).reshape(batch, self.topN, 4)
        top_n_index = top_n_cdds[:, :, -1].astype(int)
        top_n_index = torch.from_numpy(top_n_index).to(self.device)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = np.zeros([batch, self.topN, 1, 256, 512], dtype=np.float32)
        part_lc_params_raw = np.zeros([batch, self.topN, 3], dtype=np.float32)
        if return_part_data:
            part_lc_list = []
        for i in range(batch):
            lc = lc_list[i]
            selected_lc_range = set()
            if lc.shape[1] == 3:
                for j in range(self.topN):
                    [x0, x1] = top_n_cdds[i][j, 1:3].astype(int)
                    time_left = np.max([x0 - 5, 0 + 5]) / 502 * (
                        np.nanmax(lc[:, 0]) - np.nanmin(lc[:, 0])
                    ) + np.nanmin(lc[:, 0])
                    time_right = np.min([x1 - 5, 512 - 5]) / 502 * (
                        np.nanmax(lc[:, 0]) - np.nanmin(lc[:, 0])
                    ) + np.nanmin(lc[:, 0])
                    selected_lc = lc[(lc[:, 0] >= time_left) & (lc[:, 0] <= time_right)]
                    if (
                        sum(selected_lc[~np.isnan(selected_lc[:, 1])][:, 2] == 1) < 3
                        and sum(selected_lc[~np.isnan(selected_lc[:, 1])][:, 2] == 2)
                        < 6
                    ):
                        time_length = 0
                        minimum_gap = 0
                        lgamp = 0
                        if return_part_data:
                            part_lc_list.append([])
                    else:
                        if (
                            np.nanmin(selected_lc[:, 0]),
                            np.nanmax(selected_lc[:, 0]),
                        ) in selected_lc_range:
                            continue
                        else:
                            selected_lc_range.add(
                                (
                                    np.nanmin(selected_lc[:, 0]),
                                    np.nanmax(selected_lc[:, 0]),
                                )
                            )
                        time_length_1 = (
                            np.nanmax(selected_lc[selected_lc[:, 2] == 1][:, 0])
                            - np.nanmin(selected_lc[selected_lc[:, 2] == 1][:, 0])
                            if (
                                sum(
                                    (selected_lc[:, 2] == 1)
                                    & (~np.isnan(selected_lc[:, 1]))
                                )
                                > 1
                            )
                            else 0
                        )
                        time_length_2 = (
                            np.nanmax(selected_lc[selected_lc[:, 2] == 2][:, 0])
                            - np.nanmin(selected_lc[selected_lc[:, 2] == 2][:, 0])
                            if (
                                sum(
                                    (selected_lc[:, 2] == 2)
                                    & (~np.isnan(selected_lc[:, 1]))
                                )
                                > 1
                            )
                            else 0
                        )
                        time_length = max(time_length_1, time_length_2)
                        diff_t = np.diff(selected_lc[:, 0])
                        minimum_gap = np.nanmedian(diff_t[diff_t > 0])
                        amp_1 = (
                            np.nanmax(selected_lc[selected_lc[:, 2] == 1][:, 1])
                            - np.nanmin(selected_lc[selected_lc[:, 2] == 1][:, 1])
                            if (
                                sum(
                                    (selected_lc[:, 2] == 1)
                                    & (~np.isnan(selected_lc[:, 1]))
                                )
                                > 1
                            )
                            else 0
                        )
                        amp_2 = (
                            np.nanmax(selected_lc[selected_lc[:, 2] == 2][:, 1])
                            - np.nanmin(selected_lc[selected_lc[:, 2] == 2][:, 1])
                            if (
                                sum(
                                    (selected_lc[:, 2] == 2)
                                    & (~np.isnan(selected_lc[:, 1]))
                                )
                                > 1
                            )
                            else 0
                        )

                        if amp_1 == 0 and amp_2 > 0:
                            lgamp = np.log10(amp_2)
                        elif amp_1 > 0 and amp_2 == 0:
                            lgamp = np.log10(amp_1)
                        elif amp_1 > 0 and amp_2 > 0:
                            lgamp = (np.log10(amp_1) + np.log10(amp_2)) / 2
                        else:
                            lgamp = 0

                        part_imgs[i : i + 1, j] = (
                            lc_to_image_array(
                                [
                                    selected_lc[selected_lc[:, 2] == 1][:, 0],
                                    selected_lc[selected_lc[:, 2] == 2][:, 0],
                                ],
                                [
                                    selected_lc[selected_lc[:, 2] == 1][:, 1],
                                    selected_lc[selected_lc[:, 2] == 2][:, 1],
                                ],
                                figure_pixel_size=(256, 512),
                                square_size=3,
                            )
                        ).reshape(1, 256, 512)
                        part_lc_params_raw[i : i + 1, j] = np.array(
                            [time_length, minimum_gap, lgamp], dtype=np.float32
                        )
                        if return_part_data:
                            part_lc_list.append(selected_lc)
            else:
                for j in range(self.topN):
                    [x0, x1] = top_n_cdds[i][j, 1:3].astype(int)
                    time_left = (
                        np.max([x0 - 5, 0 + 5]) / 502 * (lc[-1, 0] - lc[0, 0])
                        + lc[0, 0]
                    )
                    time_right = (
                        np.min([x1 - 5, 512 - 5]) / 502 * (lc[-1, 0] - lc[0, 0])
                        + lc[0, 0]
                    )
                    selected_lc = lc[(lc[:, 0] >= time_left) & (lc[:, 0] <= time_right)]
                    if len(selected_lc[~np.isnan(selected_lc[:, 1])]) < 6 or np.all(
                        selected_lc[:, 1] == selected_lc[0, 1]
                    ):
                        time_length = 0
                        minimum_gap = 0
                        lgamp = 0
                        if return_part_data:
                            part_lc_list.append([])
                    else:
                        if (
                            np.nanmin(selected_lc[:, 0]),
                            np.nanmax(selected_lc[:, 0]),
                        ) in selected_lc_range:
                            continue
                        else:
                            selected_lc_range.add(
                                (
                                    np.nanmin(selected_lc[:, 0]),
                                    np.nanmax(selected_lc[:, 0]),
                                )
                            )
                        time_length = selected_lc[-1, 0] - selected_lc[0, 0]
                        minimum_gap = np.nanmedian(np.diff(selected_lc[:, 0]))
                        lgamp = np.log10(
                            np.nanmax(selected_lc[:, 1]) - np.nanmin(selected_lc[:, 1])
                        )
                        part_imgs[i : i + 1, j] = (
                            lc_to_image_array(
                                [selected_lc[:, 0]],
                                [selected_lc[:, 1]],
                                figure_pixel_size=(256, 512),
                                square_size=3,
                            )
                        ).reshape(1, 256, 512)
                        if return_part_data:
                            part_lc_list.append(selected_lc)
                        part_lc_params_raw[i : i + 1, j] = np.array(
                            [time_length, minimum_gap, lgamp], dtype=np.float32
                        )

        part_imgs = (
            torch.from_numpy(part_imgs)
            .view(batch * self.topN, 1, 256, 512)
            .to(self.device)
        )
        part_lc_params = (
            torch.from_numpy(part_lc_params_raw)
            .view(batch * self.topN, 3)
            .to(self.device)
        )

        _, _, _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_features = torch.cat([part_features, part_lc_params], dim=1)
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        concat_out = torch.cat([part_feature, lc_feature_out], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        if return_part_imgs:
            return [
                raw_logits,
                concat_logits,
                part_logits,
                top_n_index,
                top_n_prob,
                part_imgs,
                part_lc_params,
                top_n_cdds,
            ]
        elif return_part_data:
            return [
                concat_out,
                raw_logits,
                concat_logits,
                part_logits,
                part_lc_list,
                part_lc_params,
            ]
        elif combined_mode:
            return concat_out
        else:
            return [
                raw_logits,
                concat_logits,
                part_logits,
            ]


class ps_component(nn.Module):
    def __init__(self, topN=3, nclasses=200, device="cuda"):
        super(ps_component, self).__init__()
        self.device = device
        self.pretrained_model = resnet18(pretrained=True)
        self.expansion = 1
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ps_params_fc = nn.Sequential(
            nn.Linear(3, 256), nn.LeakyReLU(inplace=True), nn.Linear(256, 64)
        )
        self.fc = nn.Linear((512 * self.expansion + 3 + 512), nclasses)
        self.ps_proposal_net = PS_ProposalNet()
        self.topN = topN
        self.concat_net = nn.Sequential(
            nn.Linear((512 * self.expansion + 3) * (CAT_NUM + 1) + 512, nclasses),
        )
        self.partcls_net = nn.Sequential(
            nn.Linear(512 * self.expansion + 3, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, nclasses),
        )
        
    def forward(
        self,
        lc_img,
        ps_img,
        phase_img,
        lc_params,
        ps_params,
        lc_list,
        ps_list,
        return_part_imgs=False,
        return_part_data=False,
        combined_mode=False,
    ):
        ps_img = ps_img.float()
        phase_img = phase_img.float()
        ps_rpn_p2, ps_rpn_p3, ps_rpn_p4, ps_rpn_p5, ps_feature = self.pretrained_model(
            ps_img
        )

        # mask = np.array([0, 0, 0, 1, 1, 0, 1], dtype=bool)
        # ps_params_out = ps_params[:, mask]
        _, _, _, _, phase_feature = self.pretrained_model(phase_img)
        ps_feature_out = torch.cat([ps_feature, ps_params, phase_feature], dim=1)
        resnet_out = self.fc(ps_feature_out)

        batch = ps_img.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        ps_rpn_score = self.ps_proposal_net(
            ps_rpn_p2.detach(),
            ps_rpn_p3.detach(),
            ps_rpn_p4.detach(),
            ps_rpn_p5.detach(),
        )  # torch.Size([1, 1614])

        all_cdds = [
            np.concatenate(
                (
                    x.reshape(-1, 1),  # (616, 1)
                    ps_anchors(),  # (616, 2)
                    np.arange(0, len(x)).reshape(-1, 1),  # (616, 1)
                ),
                axis=1,
            )
            for x in ps_rpn_score.data.cpu().numpy()
        ]
        top_n_cdds = [
            hard_nms_ps(x, topn=self.topN, iou_thresh=0.5)
            for _, x in enumerate(all_cdds)
        ]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(int)
        top_n_index = torch.from_numpy(top_n_index).to(self.device)
        top_n_prob = torch.gather(ps_rpn_score, dim=1, index=top_n_index)

        part_imgs = np.zeros([batch, self.topN, 1, 224, 448], dtype=np.float32)
        part_phase_imgs = np.zeros([batch, self.topN, 1, 224, 224], dtype=np.float32)
        part_ps_params = np.zeros([batch, self.topN, 3], dtype=np.float32)

        if return_part_data:
            part_ps_list = []

        for i in range(batch):
            ps = ps_list[i]
            lc = lc_list[i]
            fap100 = ps_params[i, 2].item()
            selected_ps_range = set()
            for j in range(self.topN):
                [x0, x1] = top_n_cdds[i][j, 1:3]  # they are relative ratio of the image
                x0 = np.max([0, x0])
                x1 = np.min([1, x1])
                selected_ps = ps[int(x0 * len(ps)) : int(x1 * len(ps)), :]
                selected_ps_no_nan = selected_ps[~np.isnan(selected_ps[:, 1])]

                if len(selected_ps_no_nan) <= 2 or np.all(
                    selected_ps[:, 1] == selected_ps[0, 1]
                ):
                    freq_at_max_power = 0
                    period_at_max_power = 0
                    fap100 = 0
                    power_max = 0
                    if return_part_data:
                        part_ps_list.append([])
                else:
                    freq_at_max_power = selected_ps[np.argmax(selected_ps[:, 1]), 0]
                    if freq_at_max_power in selected_ps_range:
                        if return_part_data:
                            part_ps_list.append([])
                        continue
                    else:
                        selected_ps_range.add(freq_at_max_power)
                    period_at_max_power = 1 / freq_at_max_power
                    power_max = selected_ps[:, 1].max()
                    if return_part_data:
                        part_ps_list.append(selected_ps)
                if 10**fap100 > power_max and fap100 != 0:
                    part_imgs[i : i + 1, j] = np.zeros((1, 224, 448))
                    phase_img = np.zeros((1, 224, 224))
                    var_ranges = 0
                else:
                    if lc.shape[1] == 3:
                        phase_img, var_ranges = plot_phase_curve(
                            [lc[lc[:, 2] == 1][:, 0], lc[lc[:, 2] == 2][:, 0]],
                            [lc[lc[:, 2] == 1][:, 1], lc[lc[:, 2] == 2][:, 1]],
                            period=period_at_max_power
                            * (10**fap100 < power_max if fap100 != 0 else True),
                            figure_pixel_size=(224, 224),
                        )
                    else:
                        phase_img, var_ranges = plot_phase_curve(
                            [lc[:, 0]],
                            [lc[:, 1]],
                            period=period_at_max_power
                            * (10**fap100 < power_max if fap100 != 0 else True),
                            figure_pixel_size=(224, 224),
                        )
                part_phase_imgs[i : i + 1, j] = phase_img.reshape(1, 224, 224)
                part_ps_params[i : i + 1, j] = np.array(
                    [
                        freq_at_max_power,
                        period_at_max_power,
                        np.log10(var_ranges if var_ranges > 0 else 1),
                    ],
                    dtype=np.float32,
                )

        part_imgs_out = (
            torch.from_numpy(part_imgs)
            .view(batch * self.topN, 1, 224, 448)
            .to(self.device)
        )
        part_phase_imgs_out = (
            torch.from_numpy(part_phase_imgs)
            .view(batch * self.topN, 1, 224, 224)
            .to(self.device)
        )
        part_ps_params_out = (
            torch.from_numpy(part_ps_params).view(batch * self.topN, 3).to(self.device)
        )
        del part_imgs, part_phase_imgs
        _, _, _, _, part_phase_features = self.pretrained_model(
            part_phase_imgs_out.detach()
        )

        part_features = torch.cat([part_ps_params_out, part_phase_features], dim=1)
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)

        # concat_logits have the shape: B*n_classes
        concat_out = torch.cat([part_feature, ps_feature_out], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        if return_part_imgs:
            return [
                raw_logits,
                concat_logits,
                part_logits,
                top_n_index,
                top_n_prob,
                part_imgs_out,
                part_phase_imgs_out,
                part_ps_params_out,
            ]
        elif return_part_data:
            return [
                concat_out,
                raw_logits,
                concat_logits,
                part_logits,
                part_ps_list,
                part_ps_params_out,
            ]
        elif combined_mode:
            return concat_out
        else:
            return [
                raw_logits,
                concat_logits,
                part_logits,
            ]


class parameter_component(nn.Module):
    def __init__(self, nclasses=7):
        super(parameter_component, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nclasses),
        )

    def forward(self, params):
        return self.fc(params)


class combined_net(nn.Module):
    def __init__(self, nclasses=7):
        super(combined_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(
                (512 + 3) * (CAT_NUM + 1) + (512 + 3) * (CAT_NUM + 1) + 512, 2048
            ),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, nclasses),
        )

    def forward(self, lc_component_out, ps_component_out, parameter_component_out=None):
        components = [lc_component_out, ps_component_out, parameter_component_out]
        out = self.net(
            torch.cat(
                [component for component in components if component is not None], dim=1
            )
        )
        return out


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


def zero_count_loss(score, time_length_or_power):
    # make sure if time_lenght==0 then score is punished
    comp_loss = (score[time_length_or_power == 0]).mean() - (
        score[time_length_or_power != 0]
    ).mean()
    mask = torch.gt(-comp_loss, 0)
    loss = torch.where(mask, 0, comp_loss)
    return loss
