def hello():
    return "Hello World"


class TestClass:
    def __init__(self):
        self.status = True

    def find_status(self):
        return self.status

    def set_status(self, newStatus):
        self.status = newStatus

    @staticmethod
    def world():
        return "The World says Hello"
