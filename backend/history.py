"""Keep whole turns so tool calls never lose their matching result."""

import json


class History:
    def __init__(self, max_turns=6, max_characters=24_000):
        self.turns = []
        self.max_turns = max_turns
        self.max_characters = max_characters

    def add(self, messages):
        self.turns.append(messages.copy())
        while self.turns and (
            len(self.turns) > self.max_turns
            or len(json.dumps(self.turns, ensure_ascii=False)) > self.max_characters
        ):
            self.turns.pop(0)

    def messages(self):
        return [message for turn in self.turns for message in turn]
