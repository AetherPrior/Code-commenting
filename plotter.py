from sys import argv
import matplotlib.pyplot as plt
plt.style.use("seaborn")
 
 
def sliding_window_average(wsize, data):
    avg_data = []
    i, j = 0, wsize
    temp = sum(data[i:j])
 
    while j < len(data):
        avg_data.append(temp/wsize)
        temp += (data[j] - data[i])
        i += 1
        j += 1
    return avg_data
 
 
losses, wsize = [], int(argv[1])
with open("loss_values.txt", 'r') as the_file:
    data = the_file.readlines()
    for line in data:
        _, loss, = line.split('|')
        _, loss = loss.split(':')
        loss = float(loss.strip())
        losses.append(loss)
 
losses = sliding_window_average(wsize, losses)
batches = range(1, len(losses)+1)
plt.plot(batches, losses)
plt.xlabel("steps")
plt.ylabel("loss")
plt.show()