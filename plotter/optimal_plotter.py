"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""
from distutils.spawn import find_executable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
from matplotlib.legend_handler import HandlerPatch


# The LaTeX rendering is disabled since some users may not have LaTeX installed.
# plt.rc('font', **{'family': 'Computer Modern'})
# plt.rc('text', usetex=True)


class HandlerCircle(HandlerPatch):

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center = 0.75 * width, 0.5 * height
        p = patches.Circle(xy=center, radius=3)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class OptimalPairPlotter:
    """This class is used to plot the optimal pairs over time."""

    @staticmethod
    def plot_circles(optimal_choices, labels, num_col, title, dir_path, file_name, color_set,
                     b_nach=(0.5, -0.05), legend_loc=9, print_title=True, print_legend=True):

        legend_labels = []
        legend_markers = []

        for i in range(0, len(labels)):
            legend_labels.append(labels[i])
            legend_markers.append(patches.Circle((0.5, 0.5), 0.25, facecolor=color_set[i],
                                                 edgecolor=color_set[i], linewidth=1))

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        if print_title is True:
            # LaTeX rendering case. You may use the next line if you have LaTeX installed.
            # plt.title(r'Classification Model Recommendation vs.\ \textsc{' + title.title() + '} Data Stream Over Time'
            #          + '\n' + r'(\textit{from top-left corner to bottom-right corner})', fontsize=8, loc='center')
            plt.title('Classification Model Recommendation vs. ' + title.title() + ' Data Stream Over Time'
                      + '\n' + 'from top-left corner to bottom-right corner', fontsize=8, loc='center')

        ax.set_xlabel(r'$\rightarrow$')
        ax.set_ylabel(r'$\downarrow$')

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        x = 0.0125
        y = 0.9875
        l = 0.0125
        for i in range(0, len(optimal_choices)):
            ax.add_patch(patches.Circle((x, y), 0.011, color=color_set[optimal_choices[i][0]]))
            if x > 1 - 2 * l:
                x = 0.0125
                y -= 2 * l
                y = round(y, 4)
            else:
                x += 2 * l
                x = round(x, 4)

        file_path = (dir_path + file_name + "_" + title).lower()

        if print_legend is True:
            ax.legend(legend_markers, legend_labels, bbox_to_anchor=b_nach, loc=legend_loc,
                      fontsize=8, ncol=num_col, frameon=True, framealpha=1,
                      handler_map={patches.Circle: HandlerCircle()})
            fig.savefig(file_path + '_circle.png', dpi=150, bbox_inches='tight')
            fig.savefig(file_path + '_circle.pdf', dpi=150, bbox_inches='tight')
        else:
            fig_leg = pylab.figure(figsize=(13.5, 3.5), dpi=150)
            pylab.figlegend(legend_markers, legend_labels, loc='center', fontsize=14, ncol=num_col, framealpha=1,
                            handler_map={patches.Circle: HandlerCircle()})
            fig_leg.savefig(file_path + "_circle_legend.pdf", dpi=150)
            fig.savefig(file_path + "_circle.pdf", dpi=150, bbox_inches='tight')
            fig.savefig(file_path + "_circle.png", dpi=150, bbox_inches='tight')
