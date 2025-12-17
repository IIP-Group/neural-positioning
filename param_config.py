import matplotlib.pyplot as plt 
import numpy as np
import torch

class Parameter:
    def __init__(self):
        self.bandwidth = 50.056e6 # Bandwidth [Hz]
        # sc0 - scn: range of subcarriers
        self.sc0 = 508
        self.scn = 516
        self.T0 = 290 # Noise temperature [K]
        self.noise_figure_dB = 9 # Noise figure [dB]
        self.noise_figure = 10**(self.noise_figure_dB/10) # Noise figure
        self.Boltzmann_constant = 1.38e-23
        self.TX_POW_dBm = 20
        self.TX_POW = 10**(self.TX_POW_dBm/10)*1e-3
        self.N0 = (self.bandwidth*self.Boltzmann_constant*self.T0*self.noise_figure)/self.TX_POW; # Hz*J/K*K=[W] noise power normalized by transmit power so that Es = 1

    def set_UE_info(self, UE_pos, color_map=None, UE_time_stamps=None):
            self.UE_pos = UE_pos
            self.U = UE_pos.shape[0]
            if UE_time_stamps is None:
                self.UE_time_stamps = torch.arange(self.U)
            else:
                self.UE_time_stamps = UE_time_stamps
            if color_map is None:
                self.set_default_color_map()  # set the color map
            else:
                self.color_map = color_map

    def set_pos_plot_axis_limits(self, UE_pos):
        self.plot_xmin = np.floor(np.min(UE_pos[:,0]))
        self.plot_xmax = np.ceil(np.max(UE_pos[:,0]))
        self.plot_ymin = np.floor(np.min(UE_pos[:,1]))
        self.plot_ymax = np.ceil(np.max(UE_pos[:,1]))

    def set_default_color_map(self):
        # coloring of the users
        color1 = (self.UE_pos[:, 0] - np.min(self.UE_pos[:, 0])).reshape((self.U, 1)) \
                    / (np.max(self.UE_pos[:, 0]) - np.min(self.UE_pos[:, 0]))
        color2 = (self.UE_pos[:, 1] - np.min(self.UE_pos[:, 1])).reshape((self.U, 1)) \
                    / (np.max(self.UE_pos[:, 1]) - np.min(self.UE_pos[:, 1]))
        color3 = np.zeros((self.U, 1))

        self.color_map = np.concatenate((color1, color2, color3), axis=-1)

    def plot_scenario(self, passive=False, dimensions='2d',error_stats=None,ap_pos=None):
        fig = plt.figure()
        if dimensions == '3d':
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], self.UE_pos[:, 2], marker='o', c=self.color_map)
            if passive:
                ax.scatter(self.TX_pos[:, 0], self.TX_pos[:, 1], self.TX_pos[:, 2], marker='+')

            ax.scatter(self.AP_pos[:, 0], self.AP_pos[:, 1], self.AP_pos[:, 2], marker='^')

            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
        elif dimensions == '2d':
            ax = fig.add_subplot()
            ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], marker='o', c=self.color_map)
            if passive:
                ax.scatter(self.TX_pos[:, 0], self.TX_pos[:, 1], marker='+')
            
            # If ap_pos is provided, plot AP positions
            if ap_pos is not None:
                ax.scatter(ap_pos[:, 0], ap_pos[:, 1], marker='^')

            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            lims = np.array([[np.min(self.UE_pos[:,0]),np.max(self.UE_pos[:,0])], [np.min(self.UE_pos[:,1]),np.max(self.UE_pos[:,1])]]) 
            ax.set_aspect('equal')
            ax.autoscale_view()

            ax.grid(True,linewidth = 1)
            ax.set_xlim(self.plot_xmin,self.plot_xmax)
            ax.set_ylim(self.plot_ymin,self.plot_ymax)
        else:
            raise Exception('Undefined dimensions for plotting the scenario')
        
        # Add text to the plot
        if error_stats is not None:
            text_str = f"Mean err: {np.round(error_stats[0],3)}, Median err: {np.round(error_stats[1],3)},\n95\% err: {np.round(error_stats[2],3)}, Max err: {np.round(error_stats[3],3)}"
            ax.text(0.5, 1.05, text_str, transform=ax.transAxes, fontsize=12, ha='center')
        
        return ax
        
    def cdfgen(self,value,input):
        len_x = len(value)
        cdfdata = np.zeros((len_x,1))
        for idx in range(len_x):
            cdfdata[idx] = np.sum(input<value[idx])/len(input)
        return cdfdata