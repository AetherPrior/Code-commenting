import argparse
import matplotlib.pyplot as plt
plt.style.use("ggplot")

parser = argparse.ArgumentParser(description="Run the plotter")
parser.add_argument("-ws", "--window-size", type=int, default=1, help="window size for averaging")
args = parser.parse_args()
 
 
def sliding_window_average(wsize, data):
    avg_data = []
    i, j = 0, wsize
    temp = sum(data[i:j])
 
    while j < len(data):
        avg_data.append(temp/wsize)
        temp += (data[j] - data[i])
        i, j = i+1, j+1
    return avg_data
 

wsize = args.window_size
losses, bscore, mscore = [], [], []
with open("cleaned_logfile.txt", 'r') as the_file:
    for line in the_file.readlines():
        if line.startswith("[INFO] Step"):
            _, loss, = line.split('|')
            _, loss = loss.split(':')
            loss = float(loss.strip())
            losses.append(loss)
        elif line.startswith("Cumulative BLEU4"):
            _, score = line.split(':')
            bscore.append(float(score.strip()))
        elif line.startswith("Cumulative METEOR"):
            _, score = line.split(':')
            mscore.append(float(score.strip()))

# take a moving window avg of the loss, bscore and mscore
if wsize > 1:
    losses = sliding_window_average(wsize, losses)
    bscore = sliding_window_average(wsize, bscore)
    mscore = sliding_window_average(wsize, mscore)

batches = range(1, len(losses)+1)
plt.plot(batches, losses)
plt.xlabel("steps")
plt.ylabel("loss")
plt.show()

plt.plot(batches, bscore)
plt.xlabel("steps")
plt.ylabel("BLEU4")
plt.show()

plt.plot(batches, mscore)
plt.xlabel("steps")
plt.ylabel("METEOR")
plt.show()