import numpy as np

path = ['eval_dataset.npy', 'train_dataset.npy']
for p in path:
    x = np.load(p, allow_pickle=True, encoding='latin1')
    print(x.size)
    x = x.tolist()
    x = [str(i) for i in x]
    with open(f"{p[:-4]}.txt",'w') as f:
        f.write('\n'.join(x))
    #np.savetxt(f"{p[:-4]}.txt", x)
    print(f"{p} convert finished!")

