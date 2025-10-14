class Trie:
    def __init__(self):
        self.trie = {}
        
    def insert(self, word):
        node = self.trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node["*"] = True
    
    def search(self, word):
        node = self.trie
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return "*" in node
    
trie = Trie()
trie.insert("cat")
trie.insert("dog")

assert(trie.search("cat"))
assert(trie.search("dog"))
assert(not trie.search("cab"))
