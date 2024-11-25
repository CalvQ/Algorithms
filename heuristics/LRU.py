class LRUCache:

    def __init__(self, capacity: int):
        self.data = {}
        self.lru = []
        self.cap = capacity

    def get(self, key: int) -> int:
        if key in self.lru:
            self.lru.remove(key)
            self.lru = [key] + self.lru
        return self.data.get(key, -1)

    def put(self, key: int, value: int) -> None:
        if key in self.data.keys():
            self.data[key] = value
            self.lru.remove(key)
            self.lru = [key] + self.lru
            return
        if len(self.lru) == self.cap:
            del self.data[self.lru[-1]]
            self.lru = self.lru[:-1]
        self.data[key] = value
        self.lru = [key] + self.lru
