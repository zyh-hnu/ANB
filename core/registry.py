class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, name):
        key = name.lower()

        def decorator(obj):
            if key in self._items:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name):
        key = name.lower()
        if key not in self._items:
            available = ", ".join(self.available()) or "none"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}")
        return self._items[key]

    def available(self):
        return sorted(self._items.keys())


ATTACKS = Registry("attack")
MODELS = Registry("model")
