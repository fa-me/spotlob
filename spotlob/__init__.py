__name__ = "spotlob"
__version__ = "0.9.1.a dev"
__author__ = "Fabian Meyer"
__summary__ = "feature extraction and analysis pipeline for image data"
__copyright__ = "Copyright 2019, Fraunhofer ISE"
__license__ = "BSD 3-clause"
__maintainer__ = "Fabian Meyer"
__email__ = "fabian.meyer@ise.fraunhofer.de"
__url__ = "https://gitlab.cc-asp.fraunhofer.de/fmeyer/spotlob"


from .spim import Spim, SpimStage
from .defaults import default_pipeline, load_image, make_gui, show_gui
from .review import review_widget
