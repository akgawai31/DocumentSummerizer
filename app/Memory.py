# app/Memory.py

class ConversationMemory:
    def __init__(self, max_messages=10):
        self.history = []
        self.max_messages = max_messages

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_messages:
            self.history.pop(0)

    def get(self):
        return self.history

    def clear(self):
        self.history = []