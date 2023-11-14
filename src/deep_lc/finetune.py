import os
import numpy as np
from tqdm import tqdm, trange
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from .models import (
    lc_component,
    ps_component,
    combined_net,
    list_loss,
    ranking_loss,
    zero_count_loss,
)
from .dataset import data_collate_fn
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


def finetune(training_set, test_set, hyper_params, base_model, save_dir, device):
    # Load hyper parameters
    BATCH_SIZE = hyper_params["batch_size"]
    PROPOSAL_NUM = hyper_params["proposal_num"]
    LR = hyper_params["lr"]
    WD = hyper_params["wd"]
    LABELS = hyper_params["labels"]

    # Load the model
    lc_net = lc_component(topN=PROPOSAL_NUM, nclasses=len(LABELS), device=device)
    ps_net = ps_component(topN=PROPOSAL_NUM, nclasses=len(LABELS), device=device)
    combined_model = combined_net(nclasses=len(LABELS)).to(device)

    # Load the dataset
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        drop_last=False,
        collate_fn=data_collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        drop_last=False,
        collate_fn=data_collate_fn,
    )
    # Load the pretrained model
    lc_net.load_state_dict(torch.load(base_model["lc_net_state_dict"]))
    ps_net.load_state_dict(torch.load(base_model["ps_net_state_dict"]))
    combined_model.load_state_dict(torch.load(base_model["net_state_dict"]))

    # Define the optimizer
    lc_net_parameters = list(lc_net.parameters())
    ps_net_parameters = list(ps_net.parameters())
    combined_parameters = list(combined_model.parameters())
    optimizer = torch.optim.Adam(
        lc_net_parameters
        + ps_net_parameters
        + combined_parameters,
        lr=LR,
        weight_decay=WD,
    )

    # Define the loss function
    creterion = nn.CrossEntropyLoss()

    # Define the scaler for mixed precision training
    scaler = GradScaler()

    # Start training
    print("Start training.")
    for epoch in range(500):
        # Train the model
        lc_net.train()
        ps_net.train()
        combined_model.train()

        for data in tqdm(training_loader):
            (
                lc_img,
                ps_img,
                phase_img,
                lc_param,
                ps_param,
                target,
                lc_data_list,
                ps_data_list,
                parameters,
            ) = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
                data[5].to(device),
                data[6],
                data[7],
                data[8],
            )

            batch_size = lc_img.shape[0]

            optimizer.zero_grad()
            with autocast():
                (
                    ps_concat_out,
                    ps_raw_logits,
                    ps_concat_logits,
                    ps_part_logits,
                    ps_top_n_prob,
                    power_max,
                ) = ps_net(
                    None, ps_img, phase_img, None, ps_param, lc_data_list, ps_data_list
                )

                (
                    lc_concat_out,
                    lc_raw_logits,
                    lc_concat_logits,
                    lc_part_logits,
                    lc_top_n_prob,
                    top_n_t_length,
                ) = lc_net(lc_img, None, None, lc_param, None, lc_data_list, None)

                output = combined_model(
                    torch.cat([lc_concat_out, ps_concat_out, parameters], dim=1)
                )

                ps_part_loss = list_loss(
                    ps_part_logits.view(batch_size * PROPOSAL_NUM, -1),
                    target.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1),
                ).view(batch_size, PROPOSAL_NUM)
                ps_raw_loss = creterion(ps_raw_logits, target)
                ps_concat_loss = creterion(ps_concat_logits, target)
                ps_rank_loss = ranking_loss(ps_top_n_prob, ps_part_loss)
                ps_zero_count_loss = zero_count_loss(ps_top_n_prob, power_max)
                ps_partcls_loss = creterion(
                    ps_part_logits.view(batch_size * PROPOSAL_NUM, -1),
                    target.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1),
                )

                lc_part_loss = list_loss(
                    lc_part_logits.view(batch_size * PROPOSAL_NUM, -1),
                    target.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1),
                ).view(batch_size, PROPOSAL_NUM)
                lc_raw_loss = creterion(lc_raw_logits, target)
                lc_concat_loss = creterion(lc_concat_logits, target)
                lc_rank_loss = ranking_loss(lc_top_n_prob, lc_part_loss)
                lc_zero_count_loss = zero_count_loss(lc_top_n_prob, top_n_t_length)
                lc_partcls_loss = creterion(
                    lc_part_logits.view(batch_size * PROPOSAL_NUM, -1),
                    target.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1),
                )

                total_loss = (
                    lc_raw_loss
                    + lc_concat_loss
                    + lc_rank_loss
                    + lc_zero_count_loss
                    + lc_partcls_loss
                    + ps_raw_loss
                    + ps_concat_loss
                    + ps_rank_loss
                    + ps_zero_count_loss
                    + ps_partcls_loss
                ) + creterion(output, target)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Test the model
        tqdm.write("Start testing.")
        lc_net.eval()
        ps_net.eval()
        combined_model.eval()
        for data in tqdm(test_loader):
            with torch.no_grad():
                (
                    lc_img,
                    ps_img,
                    phase_img,
                    lc_param,
                    ps_param,
                    target,
                    lc_data_list,
                    ps_data_list,
                    parameters,
                ) = (
                    data[0].to(device),
                    data[1].to(device),
                    data[2].to(device),
                    data[3].to(device),
                    data[4].to(device),
                    data[5].to(device),
                    data[6],
                    data[7],
                    data[8],
                )
                batch_size = lc_img.size(0)

                lc_concat_out = lc_net(
                    lc_img, None, None, lc_param, None, lc_data_list, None, combined_model=True
                ) 
                ps_concat_out = ps_net(
                    None, ps_img, phase_img, None, ps_param, lc_data_list, ps_data_list, combined_model=True
                )

                output = combined_model(
                    torch.cat([lc_concat_out, ps_concat_out, parameters], dim=1)
                )

                _, concat_predict = torch.max(output, 1)
                test_correct += torch.sum(concat_predict.data == target.data)
        
        test_acc = float(test_correct) / len(test_set)
        tqdm.write("Epoch: {}, Test Accuracy: {:.4f}".format(epoch, test_acc))

        torch.save(
            {
                "num_classes" : len(LABELS),
                "epoch": epoch,
                "test_acc": test_acc,
                "lc_net_state_dict": lc_net.state_dict(),
                "ps_net_state_dict": ps_net.state_dict(),
                "net_state_dict": combined_model.state_dict(),
            },
            os.path.join(save_dir, "%03d.ckpt" % epoch),
        )
