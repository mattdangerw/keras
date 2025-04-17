# Hack to make Keras importable from the base of the repository.
# This will not be included in the actual packaged version of Keras.
# When Keras is actually packaged, the `api` is the toplevel package directory.
from keras.api import *  # noqa: F403
from keras.api import __version__  # Import * ignores names start with "_".

import os  # isort: skip

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

# Don't pollute namespace.
del os
