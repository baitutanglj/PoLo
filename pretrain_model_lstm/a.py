c = 0
for el, value in self.store.items():
    random.shuffle(value)
    print(len(self.store[el]))
    a = self.store[el]
    print(len(a))
    c += 1
    if c > 10:
        break