import numpy as np
import torch
from .dataset import light_curve_preprocess, fold_lightcurve, bin_timeseries
from .models import lc_component, ps_component, parameter_component, combined_net
from .config import PROPOSAL_NUM, LABELS
import matplotlib.pyplot as plt
import matplotlib


class DeepLC:
    """Base class for light curve classification."""

    def __init__(
        self,
        lc_component_model=None,
        ps_component_model=None,
        parameter_model=None,
        combined_model=None,
        device="auto",
    ) -> None:
        """Initialize the classifier.

        Parameters
        ----------
        lc_component_model : str, optional
            path to the light curve component model, by default None
        ps_component_model : str, optional
            path to the power spectrum component model, by default None
        parameter_model : str, optional
            path to the parameter model, by default None
        combined_model : str, optional
            path to the combined model, by default None
        device : str, optional
            device of the model, by default 'auto'
        """

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        models = [
            lc_component_model,
            ps_component_model,
            parameter_model,
            combined_model,
        ]
        num_models = sum(model is not None for model in models)
        if num_models == 0:
            raise ValueError("At least one model must be provided.")
        elif num_models > 1:
            raise ValueError("Only one model can be provided.")

        if combined_model:
            self.loaded_model = "Combined"
            if isinstance(combined_model, str):
                self.model_dict = torch.load(combined_model)
            elif isinstance(combined_model, dict):
                self.model_dict = combined_model
            # self.nclasses = self.model_dict["nclasses"]
            self.lc_model = lc_component(
                topN=PROPOSAL_NUM, nclasses=len(LABELS), device=self.device
            )
            self.ps_model = ps_component(
                topN=PROPOSAL_NUM, nclasses=len(LABELS), device=self.device
            )
            # TODO self.parameter_model = parameter_component()
            self.model = combined_net(nclasses=len(LABELS))
            self.lc_model.load_state_dict(self.model_dict["lc_net_state_dict"])
            self.ps_model.load_state_dict(self.model_dict["ps_net_state_dict"])
            self.model.load_state_dict(self.model_dict["net_state_dict"])
            self.lc_model.to(self.device)
            self.ps_model.to(self.device)
            self.lc_model.eval()
            self.ps_model.eval()
        elif lc_component_model:
            self.loaded_model = "LC Component"
            self.model_dict = torch.load(lc_component_model)["net_state_dict"]
            self.model = lc_component(
                topN=PROPOSAL_NUM, nclasses=len(LABELS), device=self.device
            )
            self.model.load_state_dict(self.model_dict)
        elif ps_component_model:
            self.loaded_model = "PS Component"
            self.model_dict = torch.load(ps_component_model)["net_state_dict"]
            self.model = ps_component(
                topN=PROPOSAL_NUM, nclasses=len(LABELS), device=self.device
            )
            self.model.load_state_dict(self.model_dict)
        elif parameter_model:
            self.loaded_model = "Parameter Component"
            self.model_dict = torch.load(parameter_model)["net_state_dict"]
            self.model = parameter_component(nclasses=len(LABELS))
            self.model.load_state_dict(self.model_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, 
        light_curve, 
        show_intermediate_results=False, 
        return_intermediate_data=False,
        conformal_predictive_sets=False
    ):
        """Classify the light curve data.

        Parameters
        ----------
        light_curve : (N, 2) array for time and flux,
            or (N, 3) array for time, flux, and filter,
            or (N, 4) array for time, flux, flux_error and filter
        show_intermediate_results : bool, optional
            whether to show intermediate results, by default False
        return_intermediate_data : bool, optional
            whether to return intermediate data, by default False
        conformal_predictive_sets : bool, optional
            whether to return conformal predictive sets, by default False
        """
        (
            lc_img,
            ps_img,
            folded_img,
            lc_param,
            ps_param,
            lc_data,
            ps_data,
        ) = light_curve_preprocess(light_curve)

        # move them to the corresponding device
        lc_img = lc_img.unsqueeze(0).to(self.device)
        ps_img = ps_img.unsqueeze(0).to(self.device)
        folded_img = folded_img.unsqueeze(0).to(self.device)
        lc_param = lc_param.unsqueeze(0).to(self.device)
        ps_param = ps_param.unsqueeze(0).to(self.device)
        lc_data = [lc_data]
        ps_data = [ps_data]
        light_curve = np.asanyarray(light_curve)

        if self.loaded_model == "Combined":
            if show_intermediate_results or return_intermediate_data:
                (
                    lc_concat_out,
                    lc_raw_logits,
                    lc_concat_logits,
                    lc_part_logits,
                    part_lc_list,
                    part_lc_params,
                ) = self.lc_model(
                    lc_img,
                    None,
                    None,
                    lc_param,
                    None,
                    lc_data,
                    None,
                    return_part_data=True,
                )

                (
                    ps_concat_out,
                    ps_raw_logits,
                    ps_concat_logits,
                    ps_part_logits,
                    part_ps_list,
                    part_ps_params,
                ) = self.ps_model(
                    None,
                    ps_img,
                    folded_img,
                    None,
                    ps_param,
                    lc_data,
                    ps_data,
                    return_part_data=True,
                )
                concat_logits = self.model(lc_concat_out, ps_concat_out)
                predicted_label = LABELS[torch.argmax(concat_logits, 1)]
                
                if return_intermediate_data:
                    return predicted_label, (
                        lc_raw_logits,
                        lc_concat_logits,
                        lc_part_logits,
                        part_lc_list,
                        ps_raw_logits,
                        ps_concat_logits,
                        ps_part_logits,
                        part_ps_list,
                    )
                
                figs = self.plot_intermediate_data(
                    (
                        ps_param.cpu().detach().numpy(),
                        lc_data,
                        ps_data,
                        lc_raw_logits.cpu().detach().numpy(),
                        lc_concat_logits.cpu().detach().numpy(),
                        lc_part_logits.cpu().detach().numpy(),
                        part_lc_list,
                        part_lc_params.cpu().detach().numpy(),
                        ps_raw_logits.cpu().detach().numpy(),
                        ps_concat_logits.cpu().detach().numpy(),
                        ps_part_logits.cpu().detach().numpy(),
                        part_ps_list,
                        part_ps_params.cpu().detach().numpy(),
                    )
                )
                return predicted_label, figs

            else:
                lc_concat_out = self.lc_model(
                    lc_img,
                    None,
                    None,
                    lc_param,
                    None,
                    lc_data,
                    None,
                    return_part_data=False,
                    combined_mode=True,
                )
                ps_concat_out = self.ps_model(
                    None,
                    ps_img,
                    folded_img,
                    None,
                    ps_param,
                    lc_data,
                    ps_data,
                    return_part_data=False,
                    combined_mode=True,
                )
                concat_logits = self.model(lc_concat_out, ps_concat_out)
                predicted_label = LABELS[torch.argmax(concat_logits, 1)]
                
                return predicted_label

        elif self.loaded_model == "LC Component":
            if show_intermediate_results or return_intermediate_data:
                (
                    lc_concat_out,
                    lc_raw_logits,
                    lc_concat_logits,
                    lc_part_logits,
                    part_lc_list,
                    part_lc_params,
                ) = self.model(
                    lc_img,
                    None,
                    None,
                    lc_param,
                    None,
                    lc_data,
                    None,
                    return_part_data=True,
                )
                predicted_label = LABELS[torch.argmax(lc_concat_logits, 1)]

                if return_intermediate_data:
                    return predicted_label, (
                        lc_raw_logits,
                        lc_concat_logits,
                        lc_part_logits,
                        part_lc_list)

                fig = self.plot_intermediate_data(
                    (
                        lc_data,
                        lc_raw_logits.cpu().detach().numpy(),
                        lc_concat_logits.cpu().detach().numpy(),
                        lc_part_logits.cpu().detach().numpy(),
                        part_lc_list,
                        part_lc_params.cpu().detach().numpy(),
                    )
                )
                return predicted_label, fig
            else:
                (
                    lc_raw_logits,
                    lc_concat_logits,
                    lc_part_logits,
                ) = self.model(lc_img, None, None, lc_param, None, lc_data, None)

                predicted_label = LABELS[torch.argmax(lc_concat_logits, 1)]
                return predicted_label
        elif self.loaded_model == "PS Component":
            if show_intermediate_results or return_intermediate_data:
                (
                    ps_concat_out,
                    ps_raw_logits,
                    ps_concat_logits,
                    ps_part_logits,
                    part_ps_list,
                    part_ps_params,
                ) = self.model(
                    None,
                    ps_img,
                    folded_img,
                    None,
                    ps_param,
                    lc_data,
                    ps_data,
                    return_part_data=True,
                )
                predicted_label = LABELS[torch.argmax(ps_concat_logits, 1)]
                
                if return_intermediate_data:
                    return predicted_label, (
                        ps_raw_logits,
                        ps_concat_logits,
                        ps_part_logits,
                        part_ps_list)
                
                fig = self.plot_intermediate_data(
                    (
                        lc_data,
                        ps_param.cpu(),
                        ps_data,
                        ps_raw_logits.cpu().detach().numpy(),
                        ps_concat_logits.cpu().detach().numpy(),
                        ps_part_logits.cpu().detach().numpy(),
                        part_ps_list,
                        part_ps_params.cpu().detach().numpy(),
                    )
                )
                return predicted_label, fig
            else:
                (
                    ps_raw_logits,
                    ps_concat_logits,
                    ps_part_logits,
                ) = self.model(
                    None,
                    ps_img,
                    folded_img,
                    None,
                    ps_param,
                    lc_data,
                    ps_data,
                )
                predicted_label = LABELS[torch.argmax(ps_concat_logits, 1)]
                return predicted_label

    def plot_intermediate_data(self, intermediate_data):
        if self.loaded_model == "Combined":
            (
                ps_param,
                lc_data,
                ps_data,
                lc_raw_logits,
                lc_concat_logits,
                lc_part_logits,
                part_lc_list,
                part_lc_params,
                ps_raw_logits,
                ps_concat_logits,
                ps_part_logits,
                part_ps_list,
                part_ps_params,
            ) = intermediate_data

            fig1 = plot_lc_component(
                lc_data,
                lc_raw_logits,
                lc_concat_logits,
                lc_part_logits,
                part_lc_list,
                part_lc_params,
            )
            fig2 = plot_ps_component(
                lc_data,
                ps_param,
                ps_data,
                ps_raw_logits,
                ps_concat_logits,
                ps_part_logits,
                part_ps_list,
                part_ps_params,
            )
            return fig1, fig2

        elif self.loaded_model == "LC Component":
            (
                lc_data,
                lc_raw_logits,
                lc_concat_logits,
                lc_part_logits,
                part_lc_list,
                part_lc_params,
            ) = intermediate_data

            fig = plot_lc_component(
                lc_data,
                lc_raw_logits,
                lc_concat_logits,
                lc_part_logits,
                part_lc_list,
                part_lc_params,
            )

        elif self.loaded_model == "PS Component":
            (
                lc_data,
                ps_param,
                ps_data,
                ps_raw_logits,
                ps_concat_logits,
                ps_part_logits,
                part_ps_list,
                part_ps_params,
            ) = intermediate_data

            fig = plot_ps_component(
                lc_data,
                ps_param,
                ps_data,
                ps_raw_logits,
                ps_concat_logits,
                ps_part_logits,
                part_ps_list,
                part_ps_params,
            )


def plot_lc_component(
    lc_data,
    lc_raw_logits,
    lc_concat_logits,
    lc_part_logits,
    part_lc_list,
    part_lc_params,
):
    predicted_raw_lable = LABELS[np.argmax(lc_raw_logits)]
    predicted_label = LABELS[np.argmax(lc_concat_logits)]
    predicted_part_labels = [
        LABELS[i] for i in np.argmax(lc_part_logits, axis=2).squeeze()
    ]

    lc_data = lc_data[0]

    lc_mask = np.all(part_lc_params != 0, axis=1)

    sub_lc_num = sum(lc_mask)

    indices = np.where(lc_mask)[0]

    selected_lc_list = [part_lc_list[i] for i in indices]
    selected_lc_labels = [predicted_part_labels[i] for i in indices]

    ratio_list = [1]*(int((sub_lc_num + 1) / 2) + 1)
    ratio_list[0] = 2

    fig1, ax = plt.subplots(
        int((sub_lc_num + 1) / 2) + 1,
        2,
        figsize=(8, (sub_lc_num + 1)+2),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"height_ratios": ratio_list},
    )
    if sub_lc_num == 0:
        ax = ax.reshape(-1, 2)
    gs1 = ax[0, 1].get_gridspec()
    for a in ax[0, :]:
        a.remove()
    ax_lc = fig1.add_subplot(gs1[0, :])
    colors = [matplotlib.colormaps["Dark2"](i / 6) for i in range(6)]

    # if lc_panel_num is odd number, add a blank panel
    if sub_lc_num % 2 == 1:
        ax[-1, 0].remove()
        ax[-1, 1].remove()
        gs2 = ax[-1, 1].get_gridspec()
        ax[-1, 0] = fig1.add_subplot(gs2[-1, :])

    if lc_data.shape[1] == 2:
        ax_lc.plot(lc_data[:, 0], lc_data[:, 1], "k.", ms=1)
        ax_lc.set_title(f"{predicted_label} ({predicted_raw_lable})")
        # plot vspans for selected light curves
        for i, lc in enumerate(selected_lc_list):
            ax_lc.axvspan(lc[0, 0], lc[-1, 0], color=colors[i], alpha=0.3)
            # plot selected light curves
            ax[i // 2 + 1, i % 2].plot(lc[:, 0], lc[:, 1], ".", color=colors[i], ms=1)
            # show parameters
            ax[i // 2 + 1, i % 2].set_title(f"{selected_lc_labels[i]}")
    else:
        bands = np.unique(lc_data[:, 3])
        multiband_colors = [matplotlib.colormaps["binary"](i / (len(bands)+1)) for i in range(len(bands)+1)]

        for band in bands:
            band_mask = lc_data[:, 3] == band
            ax_lc.plot(
                lc_data[band_mask, 0],
                lc_data[band_mask, 1],
                ".",
                ms=5,
                # color is a grey scale for different bands
                color=multiband_colors[int(band)],
            )
        ax_lc.set_title(f"{predicted_label} ({predicted_raw_lable})")
        # plot vspans for selected light curves
        for i, lc in enumerate(selected_lc_list):
            ax_lc.axvspan(lc[0, 0], lc[-1, 0], color=colors[i], alpha=0.3)
            # plot selected light curves
            for band in bands:
                band_mask = lc[:, 3] == band
                ax[i // 2 + 1, i % 2].plot(
                    lc[band_mask, 0],
                    lc[band_mask, 1],
                    ".",
                    color=multiband_colors[int(band)],
                    ms=5,
                )
            # show parameters
            ax[i // 2 + 1, i % 2].set_title(f"{selected_lc_labels[i]}", color=colors[i])
    
    # all the y axis are scientific notation
    ax_lc.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for a in ax.flatten():
        a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # add labels
    ax_lc.set_xlabel("Time (days)")
    ax_lc.set_ylabel("Variability")
    return fig1


def plot_ps_component(
    lc_data,
    ps_param,
    ps_data,
    ps_raw_logits,
    ps_concat_logits,
    ps_part_logits,
    part_ps_list,
    part_ps_params,
):
    predicted_raw_lable = LABELS[np.argmax(ps_raw_logits)]
    predicted_label = LABELS[np.argmax(ps_concat_logits)]
    predicted_part_labels = [
        LABELS[i] for i in np.argmax(ps_part_logits, axis=2).squeeze()
    ]
    ps_data = ps_data[0]
    lc_data = lc_data[0]

    period = ps_param[0, 1]
    # folded_lc_var = ps_param[0, 2]

    phase_list, flux_list = fold_lightcurve(lc_data, period)

    ps_mask = np.any(part_ps_params != 0, axis=1)

    ps_panel_num = sum(ps_mask)
    indices = np.where(ps_mask)[0]

    selected_ps_list = [part_ps_list[i] for i in indices]
    selected_ps_params = [part_ps_params[i] for i in indices]
    selected_ps_labels = [predicted_part_labels[i] for i in indices]

    fig2, ax = plt.subplots(
        ps_panel_num + 1,
        2,
        figsize=(6, (ps_panel_num + 1)*1.5+2),
        dpi=300,
        constrained_layout=True,
    )
    colors = [matplotlib.colormaps["Dark2"](i / 6) for i in range(6)]
    
    if lc_data.shape[1] > 2:
        bands = np.unique(lc_data[:, 3])
        multiband_colors = [matplotlib.colormaps["binary"](i / (len(bands)+1)) for i in range(len(bands)+1)]
    else:
        bands = [0]
        multiband_colors = ["k"]*2
    
    ax[0, 0].plot(ps_data[:, 0], ps_data[:, 1], "k-", ms=1)
    ax[0, 0].set_ylabel("Amp")
    ax[0, 0].set_title(f"{predicted_label}")
    for band, (phase, flux) in enumerate(zip(phase_list, flux_list)):
        # ax[0, 1].plot(phase, flux, "k.", ms=0.1)
        new_phase, new_flux = bin_timeseries(phase, flux, 512)
        ax[0, 1].plot(new_phase, new_flux, ".", ms=3, color=multiband_colors[int(band)+1] )
    ax[0, 1].set_title(f"{predicted_raw_lable} ({period:.2f} days)", loc='right', pad=0)
    

    # plot vspans for selected light curves
    for i, ps in enumerate(selected_ps_list):
        ax[0, 0].axvspan(ps[0, 0], ps[-1, 0], color=colors[i], alpha=0.5)
        # plot selected light curves
        ax[i + 1, 0].plot(ps[:, 0], ps[:, 1], "-", color=colors[i], ms=1)
        ax[i + 1, 0].set_ylabel("Amp")
        period = selected_ps_params[i][1]
        phase_list, flux_list = fold_lightcurve(lc_data, period)
        for band, (phase, flux) in enumerate(zip(phase_list, flux_list)):
            new_phase, new_flux = bin_timeseries(phase, flux, 512)
            ax[i + 1, 1].plot(new_phase, new_flux, ".", ms=3, color=multiband_colors[int(band)+1])
        # show parameters
        ax[i + 1, 1].set_title(f"{selected_ps_labels[i]} ({period:.2f} days)", loc='right', pad=0)

    # all the y-axis are scientific notation
    for a in ax.flatten():
        a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # add lables for the last pannel
    ax[-1, 0].set_xlabel("Frequency (day$^{-1}$)")
    ax[-1, 1].set_xlabel("Phase")
    return fig2
