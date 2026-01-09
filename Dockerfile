FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- Click "Commit new file"

✅ **Repo pronto!**

---

## **PARTE 2: DEPLOY NO RENDER** (5 min)

### **Passo 5:**
- Vai a https://render.com
- Click "Get Started" (se não tens conta)
- Ou "Sign in" (se já tens)
- **Usa GitHub** para fazer login (mais fácil)

### **Passo 6:**
- No dashboard do Render, click "New +"
- Escolhe "Web Service"

### **Passo 7:**
- Click "Connect" ao lado do repo `zwai-rag` que criaste
- Se não aparecer, click "Configure GitHub" e autoriza

### **Passo 8:** Configurações:
```
Name: zwai-rag
Region: Frankfurt (mais perto de Portugal)
Branch: main
Runtime: Docker
Instance Type: Free
```

- Click "Create Web Service" (ainda NÃO!)

### **Passo 9:** ANTES de criar, adiciona Environment Variables:

Click "Advanced" → scroll até "Environment Variables"

Adiciona 3 variáveis:
```
OPENAI_API_KEY = sk-proj-... (a tua key)
QDRANT_URL = https://xxxxx.eu-central.aws.cloud.qdrant.io (a tua URL)
QDRANT_API_KEY = qdr_xxxxx (a tua key)
