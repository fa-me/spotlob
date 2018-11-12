import matplotlib.pyplot as plt


class MatplotlibPreviewScreen(object):
    def __init__(self):
        pass

    def make_new(self, background_im, *args, **kwargs):
        self.fig, self.ax = plt.subplots(*args, **kwargs)
        self.bg_ax = self.ax.imshow(background_im)
        self.fg_ax = self.ax.imshow(background_im)

    def update(self, new_im):
        self.fg_ax.set_data(new_im)
        self.fig.canvas.draw_idle()
