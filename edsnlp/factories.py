from .core.registry import registry


class VirtualModule:
    def __init__(self, path=()):
        self.path = path


def make_virtual_module():
    class VirtualModule:
        pass

    return VirtualModule()


virtual = make_virtual_module()

for path in registry.factories.get_available():
    current = virtual
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(virtual, part):
            setattr(current, part, make_virtual_module())
        current = getattr(virtual, part)
    if not parts[-1].isidentifier():
        continue
    name = parts[-1]

    def make_function_getter(path, name):
        @property
        def get_function(self):
            result = registry.factories.get(path)
            delattr(type(self), name)
            setattr(self, name, result)
            return result

        return get_function

    setattr(type(current), name, make_function_getter(path, name))

__all__ = dir(virtual)
__getattr__ = virtual.__getattribute__
