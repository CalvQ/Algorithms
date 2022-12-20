'''
Problem:            Tower of Hanoi
Statement:          Given an tower of height n, how do you move 
                    the disks to a 3rd peg, making sure all disks 
                    are in increasing size.
'''


class Tower:
    def __init__(self, n):
        self.one = [x for x in range(n)]
        self.two = []
        self.three = []
        self.size = n

    def pTowers(self):
        print("---pTowers---")
        print("One: ", self.one)
        print("Two: ", self.two)
        print("Three: ", self.three)
        print()
    
    def _verify_stack(self, stack):
        return all(stack[i] < stack[i+1] for i in range(len(stack)-1))
    
    def verify(self):
        return self._verify_stack(self.one) and self._verify_stack(self.two) and self._verify_stack(self.three)

    def hanoi(self, start, end, temp, amount):
        if not self.verify():
            self.pTowers()
            assert False
        if amount == 1:
            end.append(start.pop())
        else:
            self.hanoi(start, temp, end, amount-1)
            end.append(start.pop())
            self.hanoi(temp, end, start, amount-1)

    def solve(self):
        self.hanoi(self.one, self.three, self.two, self.size)



t = Tower(4)
t.pTowers()
t.solve()
t.pTowers()
