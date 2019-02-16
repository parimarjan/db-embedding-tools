import visdom
import numpy as np
from skimage.color import hsv2rgb

# TODO: add original link

def convert_to_html(string):
    all_lines = string.split("\n")
    new_string = ""
    for s in all_lines:
        # we can apply arbitrary rules here
        if "Join" not in s:
            continue
        # get rid of random stuff
        s = s.replace(", joinType=[inner]","")
        # new_string += ("<pre>" + s + "</pre> <br>\n")
        new_string += ("<pre>" + s + "</pre> \n")
    return new_string

class Visualizer(object):
  def __init__(self, port=8097, env="main"):
    self.vis = visdom.Visdom(port=port, env=env)

class ScalarVisualizer(Visualizer):
  def __init__(self, name, ntraces=None, port=8097, env="main", opts=None):
    super(ScalarVisualizer, self).__init__(port=port, env=env)
    self.name = name
    self.time = []
    self.value = []
    self.traces = set()
    self.is_first = True

    if opts is None:
      opts = {}

    if ntraces is not None:
      if "legend" in opts.keys():
          raise ValueError("cannot specify ntraces when legend is provided, we'll infer it.")

    if ntraces is None:
      ntraces = 1

    if "legend" in opts.keys():
      ntraces = len(opts["legend"])

    if "xlabel" not in opts.keys():
      opts["xlabel"] = "epoch"
    if "ylabel" not in opts.keys():
      opts["ylabel"] = self.name
    if "title" not in opts.keys():
      opts["title"] = "{} over {}".format(opts["ylabel"], opts["xlabel"])
    self.opts = opts

    if ntraces == 1:
      y = np.nan*np.ones((1,))
    else:
      y = np.nan*np.ones((1, ntraces))

    self.vis.line(
        y,
        X=np.array([0]),
        win=self.name,
        name=name,
        opts=self.opts,
        )

  def update(self, t, v, name=None, update='append'):

    if update == "append":
        t = np.array(t)
        v = np.array(v)

        t = np.expand_dims(t, 0)
        v = np.expand_dims(v, 0)
    else:
        pass

    self.vis.line(
        v,
        X=t,
        update=update,
        win=self.name,
        name=name,
        )


class TextVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(TextVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, text):
    self.vis.text(text, win=self.name)


class HistogramVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(HistogramVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, data, numbins=30, caption=None):
    self.vis.histogram(
        data,
        opts={'numbins': numbins,
              'caption':caption,
              'title': "{}".format(self.name)},
        win=self.name)


class ImageVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(ImageVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, image, caption=None):
    self.vis.images(
        image,
        opts={
          'title': "{}".format(self.name),
          'jpgquality': 100,
          'caption': caption,
          },
        win=self.name)


class BatchVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(BatchVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, images, per_row=8, caption=None):
    self.vis.images(
        images,
        nrow=per_row,
        opts={
          'title': "{}".format(self.name),
          'jpgquality': 100,
          'caption': caption,
          },
        win=self.name)


class ScatterVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(ScatterVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, x, title="", xlabel="x", ylabel="y",
             color=None, markersize=None):
    plot_update = None
    if self.vis.win_exists(self.name, env=self.vis.env):
      plot_update = False

    plot_update = None
    self.vis.scatter(
        X=x,
        update=plot_update,
        opts={
          'title': title,
          'xlabel': xlabel,
          'ylabel': ylabel,
          'markercolor': color,
          'markersize': markersize,
          },
        win=self.name)


class GraphVisualizer(Visualizer):
  def __init__(self, name, port=8097, env="main"):
    super(GraphVisualizer, self).__init__(port=port, env=env)
    self.name = name

  def update(self, t, v, legend=None):
    self.vis.line(
        X=np.array(t),
        Y=np.array(v),
        opts={
          'title': "{}".format(self.name),
          'legend': legend
          },
        win=self.name)
