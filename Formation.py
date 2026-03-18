
class Formation:

    LITHOLOGY_LIST = [

    ]

    def __init__(self, age, lithology):
        self.age = age
        self.lithology = self.onehot_lithology(lithology)

    def onehot_lithology(self, lithology):

        pass
