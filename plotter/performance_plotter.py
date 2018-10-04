"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import pylab
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


# The LaTeX rendering is disabled since some users may not have LaTeX installed.
# plt.rc('font', **{'family': 'Computer Modern'})
# plt.rc('text', usetex=True)


class Plotter:
    """This class is used to plot, for example, the error-rate of a learning algorithm."""

    @staticmethod
    def plot_single(learner_name, performance_array, y_title,
                    project_name, dir_path, file_name, y_lim, legend_loc, zip_size, colour="ORANGERED"):

        x = []
        y = []
        for i in range(0, len(performance_array)):
            if i % zip_size == 0 or i == len(performance_array) - 1:
                x.append((i / len(performance_array)) * 100)
                y.append(performance_array[i])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # LaTeX rendering case. You may use the next line if you have LaTeX installed.
        # ax.set_title(r'\textsc{' + project_name.title() + '}', fontsize=14)
        ax.set_title(project_name.title(), fontsize=14)

        ax.set_xlim(0, 100)
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_xlabel('Percentage of Instances', fontsize=14)
        ax.set_ylabel(y_title, fontsize=14)
        ax.grid()

        ax.plot(x, y, color=colour, linewidth=1.2, label=learner_name)

        leg = ax.legend(fontsize=14, loc=legend_loc, framealpha=0.9)
        for leg_obj in leg.legendHandles:
            leg_obj.set_linewidth(2.0)

        # LaTeX rendering case. You may use the next line if you have LaTeX installed.
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '\%'))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '%'))

        file_path = (dir_path + file_name + "_" + y_title).lower()

        plt.tight_layout()
        plt.savefig(file_path + ".pdf", dpi=150)
        plt.savefig(file_path + ".png", dpi=150)

    @staticmethod
    def plot_multiple(pairs_names, num_instances, performances_array, y_title,
                      project_name, dir_path, file_name, y_lim, b_anch, legend_loc, col_num, zip_size,
                      color_set, z_orders, print_legend=True):

        x = []
        y = []
        for i in range(0, len(pairs_names)):
            y.append([])
        for i in range(0, num_instances):
            if i % zip_size == 0 or i == num_instances - 1:
                x.append((i / num_instances) * 100)
                for j in range(0, len(pairs_names)):
                    y[j].append(performances_array[j][i])

        fig = plt.figure()

        ax = fig.add_subplot(111)

        # LaTeX rendering case. You may use the next line if you have LaTeX installed.
        # ax.set_title(r'\textsc{' + project_name.title() + '}', fontsize=14)
        ax.set_title(project_name.title(), fontsize=14)

        ax.set_xlim(0, 100)
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_ylabel(y_title, fontsize=14)
        ax.set_xlabel("Percentage of Instances", fontsize=14)
        ax.grid()

        for i in range(0, len(pairs_names)):
            ax.plot(x, y[i], label=pairs_names[i], color=color_set[i], linewidth=1.2, zorder=z_orders[i])

        # LaTeX rendering case. You may use the next line if you have LaTeX installed.
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '\%'))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '%'))

        file_path = (dir_path + file_name + "_" + y_title).lower()

        if print_legend is True:
            leg = ax.legend(bbox_to_anchor=b_anch, loc=legend_loc, ncol=col_num, fontsize=8, framealpha=1)
            for leg_obj in leg.legendHandles:
                leg_obj.set_linewidth(1)
            fig.savefig(file_path + ".pdf", dpi=150, bbox_inches='tight')
            fig.savefig(file_path + ".png", dpi=150, bbox_inches='tight')
        else:
            fig_leg = pylab.figure(figsize=(13.5, 3.5), dpi=150)
            leg = pylab.figlegend(*ax.get_legend_handles_labels(), loc='center', ncol=col_num, fontsize=14
                                  , framealpha=1)
            for leg_obj in leg.legendHandles:
                leg_obj.set_linewidth(3.0)
            fig_leg.savefig(file_path + "_legend.pdf", dpi=150)
            fig_leg.savefig(file_path + "_legend.png", dpi=150)
            fig.savefig(file_path + ".pdf", dpi=150, bbox_inches='tight')
            fig.savefig(file_path + ".png", dpi=150, bbox_inches='tight')

    @staticmethod
    def plot_single_ddm_points(learner_name, drift_points, project_name, dir_path, file_name, colour="ORANGERED"):

        fig = plt.figure(figsize=(10, 0.75))

        y = drift_points
        x = []
        y_ = []
        for j in range(0, len(y)):
            if y[j] == 1:
                x.append((j / len(y)) * 100)
                y_.append(1)

        ax = plt.subplot(111)

        # LaTeX rendering case. You may use the next line if you have LaTeX installed.
        # ax.set_title(r'\textsc{' + project_name.title() + '}' + " vs.\ " + learner_name, fontsize=14, loc='left')
        ax.set_title(project_name.title() + " vs. " + learner_name, fontsize=14, loc='left')

        ax.scatter(x, y_, 30, edgecolors=colour, color=colour, label=learner_name)
        ax.set_xlim(0, 100)
        ax.set_ylim(0.95, 1.05)

        # LaTeX rendering case. You may use the next line if you have LaTeX installed.
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '\%'))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '%'))

        ax.xaxis.set_tick_params(labelsize=9)
        ax.yaxis.set_visible(False)

        file_path = (dir_path + file_name + "_drifts").lower()

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig(file_path + ".pdf", dpi=150, bbox_inches='tight')
        fig.savefig(file_path + ".png", dpi=150, bbox_inches='tight')

    @staticmethod
    def plot_multi_ddms_points(pairs_names, d_lists, project_name, dir_path, file_name, color_set):

        num_subplots = len(pairs_names)

        fig = plt.figure(figsize=(10, 0.75 * num_subplots))

        for i in range(0, num_subplots):
            y = d_lists[i]
            x = []
            y_ = []
            for j in range(0, len(y)):
                if y[j] == 1:
                    x.append((j / len(y)) * 100)
                    y_.append(1)

            ax = plt.subplot(num_subplots, 1, i + 1)

            # LaTeX rendering case. You may use the next line if you have LaTeX installed.
            # ax.set_title(r'\textsc{' + project_name.title() + '}' + " vs.\ " + pairs_names[i],
            #             fontsize=14, loc='left')
            ax.set_title(project_name.title() + " vs. " + pairs_names[i], fontsize=14, loc='left')

            ax.scatter(x, y_, 30, edgecolors=color_set[i], color=color_set[i], label=pairs_names[i])
            ax.set_xlim(0, 100)
            ax.set_ylim(0.95, 1.05)

            # LaTeX rendering case. You may use the next line if you have LaTeX installed.
            # ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '\%'))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda ix, _: '%1.0f' % ix + '%'))

            ax.xaxis.set_tick_params(labelsize=9)
            if i < len(pairs_names) - 1:
                ax.set_xticklabels([])
            ax.yaxis.set_visible(False)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        file_path = (dir_path + file_name + "_drifts").lower()

        fig.savefig(file_path + ".pdf", dpi=150, bbox_inches='tight')
        fig.savefig(file_path + ".png", dpi=150, bbox_inches='tight')
