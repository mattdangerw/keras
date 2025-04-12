from keras.src.backend.common import global_state


class CloneScope:
    """Scope to indicate model cloning."""

    def __enter__(self):
        self.original_scope = get_clone_scope()
        global_state.set_global_attribute("clone_scope", self)
        return self

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("clone_scope", self.original_scope)


def in_clone_scope():
    return global_state.get_global_attribute("clone_scope") is not None


def get_clone_scope():
    return global_state.get_global_attribute("clone_scope")
