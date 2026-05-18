from enum import Enum


class TrafficLightState(Enum):
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


class DirectionType(Enum):
    STOP = 0
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3


class CollisionType(Enum):
    NO_COLLISION = 0
    VEHICLE = 1
    WALKER = 2
    STATIC = 3
    OTHER = 4


class LaneMarkingType(Enum):
    SOLID = "Solid"  # 严格禁止跨越
    BROKEN = "Broken"  # 安全时可以跨越
    SOLID_SOLID = "SolidSolid"  # 禁止跨越
    BROKEN_SOLID = "BrokenSolid"  # 一侧可变道
    SOLID_BROKEN = "SolidBroken"  # 一侧可变道
    UNKNOWN = "Unknown"
