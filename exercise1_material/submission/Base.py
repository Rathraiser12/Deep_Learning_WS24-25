# Layers/Base.py


class BaseLayer:
    def __init__(self):
        self.trainable = False
        # Optional case to  initialize other members like weights
        self.weights = None
