from dataclasses import dataclass

@dataclass
class DynamicObject(object):
    initialVelocity: float
    initialX: float
    initialY: float
    initialOrientation: float

    pass

@dataclass
class Ego(DynamicObject):
    pass
