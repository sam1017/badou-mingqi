class Tree:
    def __init__(self, val = "#", left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
        self.father = None
        self.count = 0
        self.allChild = []

    def resetAllChild(self):
        self.allChild = []

    def getAllChild(self):
        if self.left is not None:
            if self.left.count > 0:
                self.left.getAllChild()
            else:
                #print("self.allChild.append self.left: ", self.left.val)
                self.printTree(self.left)
        if self.right is not None:
            if self.right.count > 0:
                self.right.getAllChild()
            else:
                #print("self.allChild.append self.right: ", self.right.val)
                self.printTree(self.right)

    def printTree(self, tree):
        words = []
        while tree is not None:
            #print("tree.val: ", tree.val)
            words.insert(0, tree.val)
            if tree.father is not None:
                tree = tree.father
            else:
                tree = None
        print(words)





