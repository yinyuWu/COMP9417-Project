class LabelEncoder():
    def __init__(self, labels):
        self.label_list = list(labels)
        self.label_set = list(set(self.label_list))
        self.trans_list = []
        self.map_list = []
        for i in range(len(self.label_set)):
            self.trans_list.append(i)
            self.map_list.append(self.label_set[i])
        
    def inverse(self, trans_number):
        return self.map_list[trans_number]
    
    def transform(self):
        t = []
        for each in self.label_list:
            index = self.find_index(each)
            t.append(index)
        return t
    
    def find_index(self, data):
        for i in range(len(self.label_set)):
            if (self.label_set[i] == data):
                return i
        return -1


def main():
    labels = ['bg', 'ba', 'ba', 'bg', 'ba', 'bg', 'bg', 'ba', 'bg']
    le = LabelEncoder(labels)
    trans_labels = le.transform()
    print(trans_labels)
    for each in trans_labels:
        print(le.inverse(each))

if __name__ == "__main__":
    main()