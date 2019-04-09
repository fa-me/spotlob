import matplotlib.pyplot as plt


class MatplotlibPreviewScreen(object):
    def __init__(self):
        self.fg_ax = None

    def make_new(self, background_im, *args, **kwargs):
        kwargs.update({'num': 'spotlob'})
        self.fig = plt.figure(*args, **kwargs)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.fig.add_axes(self.ax)
        # self.bg_ax = self.ax.imshow(background_im)
        self.fg_ax = self.ax.imshow(background_im, cmap="gray")
        self.ax.set_axis_off()

    def update(self, new_im):
        """redraw the image if foreground axis is defined"""
        if self.fg_ax:
            self.fg_ax.set_data(new_im)
            self.fig.canvas.draw_idle()
