import random


class Memory(object):
    """
    Using a list as the data structure for memory instead of deque. This has O(n) append performance
    when the structure is full, but O(1) for accessing elements compared to deque's O(1) append, but
    O(n) for accessing elements. Since the memory is sampled from more than appending, it makes sense
    to prioritize sample performance.
    """

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.items = []

    def append(self, item):
        self.items.append(item)
        if len(self.items) > self.maxlen:
            self.items.pop(0)

    def sample(self, n):
        n = min(n, len(self.items))
        return random.sample(self.items, n)
