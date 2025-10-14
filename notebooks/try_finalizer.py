import weakref


class TestClass:
    count = 0
    def __init__(self):
        TestClass.count += 1
        self.idx = TestClass.count
        weakref.finalize(self, self._cleanup, self.idx)
        print('__init__:', self.idx)

    @staticmethod
    def _cleanup(name):
        print('_cleanup:', name)

def func():
    return TestClass()

if __name__ == '__main__':
    t1 = TestClass()
    print('after t1')

    TestClass()
    print('after anonymous object')

    func()
    print('after func')
