import inspect


class Component(object):
    context_provides = []

    def pipeline_init(self, *args):
        pass

    def train(self, *args):
        pass

    def process(self, *args):
        pass

    def persist(self, model_dir):
        pass

    def cache_key(self):
        return None

    @classmethod
    def load(cls, *args):
        return cls()

    def pipeline_init_args(self):
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.pipeline_init).args)

    def train_args(self):
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.train).args)

    def process_args(self):
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.process).args)

    @classmethod
    def load_args(cls):
        return filter(lambda arg: arg not in ["cls"], inspect.getargspec(cls.load).args)

