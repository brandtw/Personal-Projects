class MyClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'MyClass({self.x}, {self.y})'

my_obj = MyClass(1, 2)
print(repr(my_obj))