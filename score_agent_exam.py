"""Score the VS Code agent's exam: outputs/exam.csv vs artifacts/exam_gt.csv."""
import pandas as pd, os
DATA = r"C:\Users\ASUS\Desktop\trendyol"
out = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_judge", "outputs", "exam.csv"))
gt = pd.read_csv(os.path.join(DATA, "artifacts", "exam_gt.csv"))
m = gt.merge(out, on="id", suffixes=("_gt", "_agent"))
print(f"eslesen: {len(m)}/200")
pos = m[m.label_gt == 1]; neg = m[m.label_gt == 0]
print(f"RECALL: {pos.label_agent.mean():.3f}   FPR: {neg.label_agent.mean():.3f}")
print("hedef: RECALL>=0.95 FPR<=0.05  (mini: 0.94/0.06, gpt-4o: 0.99/0.02)")
