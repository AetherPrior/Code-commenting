import argparse
import matplotlib.pyplot as plt
plt.style.use("ggplot")

parser = argparse.ArgumentParser(description="Run the plotter")
parser.add_argument("-f", "--file-name", type=str, help="filename to be plotted")
args = parser.parse_args()
 
bscore, mscore = [], []
with open(args.file_name, 'r') as the_file:
    for line in the_file.readlines():
        if line.startswith("Cumulative BLEU4"):
            _, score = line.split(':')
            bscore.append(float(score.strip()))
        elif line.startswith("Cumulative METEOR"):
            _, score = line.split(':')
            mscore.append(float(score.strip()))

print(f"[BLEU4] max: {max(bscore)}, min: {min(bscore)}, avg: {sum(bscore)/len(bscore)}")
print(f"[METEOR] max: {max(mscore)}, min: {min(mscore)}, avg: {sum(mscore)/len(bscore)}")

plt.boxplot(bscore)
plt.title("BLEU4 score distribution")
plt.show()

plt.boxplot(mscore)
plt.title("METEOR score distribution")
plt.show()