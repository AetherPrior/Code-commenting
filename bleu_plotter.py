import matplotlib.pyplot as plt
plt.style.use("seaborn")

steps, loss, bleu1, bleu2, bleu3, bleu4 = [], [], [], [], [], []

with open("hist.txt" , 'r') as the_file:
    for line in the_file.readlines():
        if line.startswith("[INFO] Batch"):
            v = line.split('|')[1].split(':')[1].strip()
            loss.append(float(v))
        elif line.startswith("Cumulative 1-gram"):
            v = line.split(':')[1].strip()
            bleu1.append(float(v))
        elif line.startswith("Cumulative 2-gram"):
            v = line.split(':')[1].strip()
            bleu2.append(float(v))
        elif line.startswith("Cumulative 3-gram"):
            v = line.split(':')[1].strip()
            bleu3.append(float(v))
        elif line.startswith("Cumulative 4-gram"):  
            v = line.split(':')[1].strip()
            bleu4.append(float(v)) 


steps = [100*i for i in range(len(loss))]
plt.plot(steps, loss)
plt.xlabel("steps")
plt.ylabel("loss")
plt.title("Loss vs Steps")
plt.show()

plt.plot(steps, bleu1, color="r", label="bleu-1 scores")
plt.plot(steps, bleu2, color="c", label="bleu-2 scores")
plt.plot(steps, bleu3, color="m", label="bleu-3 scores")
plt.plot(steps, bleu4, color="y", label="bleu-4 scores")
plt.legend(["bleu1", "bleu2", "bleu3", "bleu4"], loc="best")
plt.title("Bleu scores vs Steps")
plt.show()

