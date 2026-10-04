# Ripristino head_swap.py

Il file sul remote e' stato troncato da un push precedente.
Sostituiscilo con la versione completa (488 righe, v3.2, hair=False).

```bash
# 1) Scarica lo zip head_swap_v32_complete.zip da Grok
# 2) Nella root del repo:
cd /Users/francescosaf/Workspace/PersonalPrj/ai-face-swap-grok
cp /path/to/head_swap.py .
git add head_swap.py
git commit -m "fix: restore complete head_swap.py v3.2 (hair transfer OFF)"
git push origin fix/no-floating-hair

# 3) Rilancia
source dlfolder/venv312/bin/activate
python headless_faceswap.py --video "..." --photo foto3.jpg foto1.jpg foto2.jpg --map 0,1,2 --quality high --limit-frames 30
```

Controlli sul file corretto:
- ~488 righe
- `"hair": False` nei tre profili QUALITY
- nessun testo PLACEHOLDER / LOADING_FROM_FILE
- `self.hair_transfer = bool(q.get("hair", False))`
