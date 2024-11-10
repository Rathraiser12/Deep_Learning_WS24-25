# Layers/Base.py


class BaseLayer:
    def __init__(self):
        self.trainable = False
        # Optionally, initialize other members like weights
        self.weights = None
