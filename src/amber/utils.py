'''Helper functions'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_if_exists(filepath, logger):
    '''
    Returns input filepath if exists, raises error if path not found
    
    Parameters
    -------------------------------------
    filepath: str or pathlib.Path
    
    logger: logging.logger
    '''
    filepath = Path(filepath).expanduser()
    
    if not filepath.exists():
        msg = (
            f"Path not found at {filepath}!"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
        
    return filepath
        
        
def plot_batch(waves=None, labels=None, only_waves=False, 
               dataloader=None, batch_idx=0, 
               seq_pick=True, picklines=True, no_noise=False,
               argmax_pick=False, argmax_color=['red', 'blue', 'green']):
    '''
    Plotting function for visual debugging (compatible with both Seisbench and
    AMBER).
    
    Parameters
    -------------------------------------------
    waves: np.ndarray or torch.tensor
        (n_batch, n_sta, n_chnl, n_dp) or 
        (n_batch, n_chnl, ndp)
        
    labels: np.ndarray or torch.tensor
        (n_batch, n_sta, n_cls, n_dp) or
        (n_batch, n_cls, n_dp)
        
    dataloader: torch.Dataloader
    
    only_waves: bool
        plot only waves (no labels)
        
    batch_idx: int
        batch index to plot when dataloader is used
        
    seq_pick: bool
        plot class score sequences for P and S (class 0 and 1),
        suitable for probability or logit outputs
        
    picklines: bool
        draw vertical lines at positions where class score 
        exceeds a fixed threshold (0.98) when seq_pick is True
        (assumes probability values [0-1])
        
    no_noise: bool
        suppress visualization of noise (class 2) when seq_pick 
        is True
        
    argmax_pick: bool
        plot labels as coloured segments corresponding
        to most probable class (class-agnostic)
        
    argmax_colors: list of str
        colors to use for each class when argmax_pick is True
    '''
    if dataloader is not None:
        for i, data in enumerate(dataloader):
            if i == batch_idx:
                break

        if isinstance(data, dict):    # For Seisbench-compatibility
            waves = data.get("X", data.get("waves"))
            labels = data.get("y", data.get("labels"))
        elif isinstance(data, (list, tuple)):
            waves = data[0]
            labels = data[1] if len(data) > 1 else None
        else:
            raise TypeError("Unsupported dataloader output type.")
            
    elif any(x is None for x in [waves, labels]) and not only_waves:
        print('Input data insufficient for generating plots!')
        return None    

    waves = np.asarray(waves)
    labels = np.asarray(labels)
    
    if waves is not None and only_waves:
        labels = np.zeros_like(waves)
        labels[:, :, 2] = 1

    print(waves.shape, labels.shape)

    if len(waves.shape) == 3:
        waves = np.expand_dims(waves, axis=1)
        labels = np.expand_dims(labels, axis=1)
        
    if argmax_pick:
        labels = np.expand_dims(np.argmax(labels, axis=2), axis=2)

    nbatch = waves.shape[0]
    nstation = waves.shape[1]


    total_subplots = nbatch * nstation + nbatch - 1
    fig, axes = plt.subplots(total_subplots, 1, figsize=(14, 0.5 * total_subplots + nbatch), 
                             sharex=True, gridspec_kw={'hspace': 0})
    
    if total_subplots == 1:
        axes = [axes]

    ax_idx = 0

    for j, event in enumerate(waves):
        for i, station_data in enumerate(event):
            ax_w = axes[ax_idx]  
            label_data = labels[j][i].squeeze()

            for chnl in range(3):              
                ax_w.plot(station_data[chnl, :])

            if not only_waves:
                if argmax_pick:
                    ylim = ax_w.get_ylim()
                    for k in range(len(argmax_color)):
                        ax_w.fill_between(
                            np.arange(label_data.size), 
                            ylim[0], ylim[1], 
                            where=(label_data == k), color=argmax_color[k], alpha=0.3
                        )
                elif seq_pick:
                    ax_w.plot(labels[j][i][0]*station_data[:3].max())
                    ax_w.plot(labels[j][i][1]*station_data[:3].max())
                    if picklines:
                        for pick_idx in np.where(labels[j][i][0] > 0.98)[0]:
                            ax_w.axvline(x=pick_idx, color='red', linestyle='--')
                        for pick_idx in np.where(labels[j][i][1] > 0.98)[0]:
                            ax_w.axvline(x=pick_idx, color='blue', linestyle='--')
                    if labels.shape[-2] == 3 and not no_noise:
                        ax_w.plot(labels[j][i][2]*station_data[:3].max())
                else:
                    for n_pick in range(labels.shape[-1]):
                        ax_w.axvline(x=labels[j][i][0][n_pick], color='red', linestyle='--')
                        ax_w.axvline(x=labels[j][i][1][n_pick], color='blue', linestyle='--')

            ax_w.tick_params(labelbottom=True)
            ax_idx += 1

        if j < nbatch - 1:
            axes[ax_idx].axis('off')
            ax_idx += 1

    plt.tight_layout(pad=1.0)

    plt.show()

    return None