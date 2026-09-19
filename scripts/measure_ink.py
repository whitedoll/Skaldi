"""글자 줄 영역의 잉크 밀도(검은 픽셀 비율). 굵은 폰트일수록 높다.
distance transform 은 작은 글자에서 값이 √2 격자로 양자화돼 폰트 굵기를 구분 못 하므로
이 지표를 함께 본다."""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(sys.argv[0]).parent))
from measure_style import bubbles, lines_in

def run(label, files):
    dens = []
    for f in files:
        g = np.array(Image.open(f).convert("L"))
        H = g.shape[0]
        for b in bubbles(g, H):
            ls = lines_in(g, b)
            if len(ls) < 2:
                continue
            hs = [c - a + 1 for a, c in ls]
            gs = [ls[i][0] - ls[i-1][0] for i in range(1, len(ls))]
            if np.median(hs) > H * 0.035 or np.median(gs) > H * 0.045:
                continue
            x, y, w, h = b
            pad = max(2, int(min(w, h) * 0.10))
            sub = g[y+pad:y+h-pad, x+pad:x+w-pad]
            for a, c in ls:
                row = sub[a:c+1]
                if row.size < 100:
                    continue
                xs = np.where((row < 128).any(axis=0))[0]
                if len(xs) < 5:
                    continue
                seg = row[:, xs.min():xs.max()+1]
                dens.append(float((seg < 128).mean()))
    print(f"[{label}] 글자 줄 {len(dens)}개 · 잉크 밀도 중앙값 {np.median(dens)*100:.1f}%"
          f"  (25%~75%: {np.percentile(dens,25)*100:.1f}~{np.percentile(dens,75)*100:.1f})")

if __name__ == "__main__":
    label, d = sys.argv[1], Path(sys.argv[2])
    files = [d / n for n in sys.argv[3:]]
    run(label, [f for f in files if f.exists()])
