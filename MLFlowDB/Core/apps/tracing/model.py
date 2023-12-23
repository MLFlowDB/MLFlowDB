import mongoengine as me

class ProvOnlineResult(me.Document):
    nodes = me.ListField()
    edges = me.ListField()

from enum import auto, Enum


class TracingForward(Enum):
    UP_STREAM = "up"
    DOWN_STREAM = "down"
    TIMELINE = "history"
    ALL = "all"
    ALL_NO_HISTORY_SPACE = "all_history_no_up_and_down"
    SPACE = "up_and_down"


    def is_upstream(self):
        if self in [TracingForward.UP_STREAM,TracingForward.SPACE,TracingForward.ALL,TracingForward.ALL_NO_HISTORY_SPACE]:
            return True
        return False

    def is_downstream(self):
        if self in [TracingForward.DOWN_STREAM,TracingForward.SPACE,TracingForward.ALL,TracingForward.ALL_NO_HISTORY_SPACE]:
            return True
        return False

    def is_timeline(self):
        if self in [TracingForward.TIMELINE,TracingForward.ALL,TracingForward.ALL_NO_HISTORY_SPACE]:
            return True
        return False

    def is_timeline_all(self):
        if self in [TracingForward.ALL]:
            return True
        return False

    @classmethod
    def get_values(cls):
        return [member.value for member in cls]